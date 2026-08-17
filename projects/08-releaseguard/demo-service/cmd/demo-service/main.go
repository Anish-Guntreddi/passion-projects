// Command demo-service is the small instrumented HTTP service ReleaseGuard
// deploys as its canary target. It has no business logic of its own -- its
// only job is to expose /health, /version, /work and /metrics with
// env-var-controlled fault injection so the controller (Phase 4+) has
// something real to evaluate.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"releaseguard/demo-service/internal/config"
	"releaseguard/demo-service/internal/server"
	"releaseguard/demo-service/internal/telemetry"
	"releaseguard/demo-service/internal/version"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg, err := config.Load(os.Getenv)
	if err != nil {
		// An invalid configuration must fail loudly at startup rather than
		// silently falling back to a guessed value -- the same principle
		// (never treat missing/bad evidence as fine) applies to the demo
		// service's own boot sequence, not just the controller's decisions.
		logger.Error("invalid configuration", "error", err)
		os.Exit(1)
	}

	info := version.Current()

	tp, shutdownTracing, err := telemetry.NewTracerProvider(context.Background(), telemetry.Config{
		ServiceName:          "releaseguard-demo-service",
		ServiceVersion:       info.Version,
		ReleaseTrack:         string(cfg.ReleaseTrack),
		ReleaseVersion:       cfg.ReleaseVersion,
		OTLPExporterEndpoint: cfg.OTLPExporterEndpoint,
		SampleRatio:          cfg.OTELTracesSampleRatio,
	})
	if err != nil {
		// Same principle as config validation above: a tracing setup that
		// can't be trusted must fail loudly at startup, not silently run
		// with a half-configured exporter.
		logger.Error("failed to initialize tracing", "error", err)
		os.Exit(1)
	}

	srv := server.New(cfg, logger, server.WithTracerProvider(tp))

	httpServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	logger.Info("starting demo-service",
		"port", cfg.Port,
		"track", cfg.ReleaseTrack,
		"release_version", cfg.ReleaseVersion,
		"image_version", info.Version,
		"git_commit", info.GitCommit,
		"build_time", info.BuildTime,
		"otlp_endpoint", cfg.OTLPExporterEndpoint,
	)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		logger.Info("shutdown signal received, draining connections")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			logger.Error("graceful shutdown failed", "error", err)
			os.Exit(1)
		}
	}

	// Flush any spans still buffered in the batch processor. This is
	// best-effort: a failed flush (e.g. the collector is also mid-restart)
	// is worth logging but must not turn an otherwise-clean shutdown into
	// a failed one -- losing the tail of trace data is a much smaller
	// problem than a rolling deployment's pods failing to terminate.
	tracingShutdownCtx, tracingCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer tracingCancel()
	if err := shutdownTracing(tracingShutdownCtx); err != nil {
		logger.Warn("tracing shutdown did not complete cleanly", "error", err)
	}

	logger.Info("demo-service stopped")
}
