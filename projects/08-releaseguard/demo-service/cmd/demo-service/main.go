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

	srv := server.New(cfg, logger)

	httpServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	info := version.Current()
	logger.Info("starting demo-service",
		"port", cfg.Port,
		"track", cfg.ReleaseTrack,
		"release_version", cfg.ReleaseVersion,
		"image_version", info.Version,
		"git_commit", info.GitCommit,
		"build_time", info.BuildTime,
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

	logger.Info("demo-service stopped")
}
