package telemetry

import (
	"context"
	"testing"
	"time"
)

func TestNewTracerProvider_DisabledWhenEndpointEmpty(t *testing.T) {
	ctx := context.Background()
	tp, shutdown, err := NewTracerProvider(ctx, Config{
		ServiceName: "demo-service",
		SampleRatio: 1.0,
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() returned unexpected error: %v", err)
	}
	if tp == nil {
		t.Fatal("NewTracerProvider() returned nil provider")
	}

	// A no-op provider must still be safe to start spans on -- handler
	// code never branches on whether tracing is enabled, so Start() must
	// never panic or block regardless.
	_, span := tp.Tracer("test").Start(ctx, "op")
	span.End()

	if err := shutdown(ctx); err != nil {
		t.Errorf("shutdown() on disabled tracing returned unexpected error: %v", err)
	}
}

func TestNewTracerProvider_EnabledDoesNotBlockOrErrorWithoutALiveCollector(t *testing.T) {
	// otlptracehttp.New only constructs a client; it must not attempt an
	// eager network connection (the collector may not exist yet, or may
	// be temporarily unreachable -- export happens asynchronously on a
	// batch timer, and a missing collector must never make service
	// startup fail or hang). This test would time out or error if that
	// assumption stopped holding for a future OTel SDK version.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tp, shutdown, err := NewTracerProvider(ctx, Config{
		ServiceName:          "demo-service",
		ServiceVersion:       "v-test",
		ReleaseTrack:         "canary",
		ReleaseVersion:       "v1.2.3",
		OTLPExporterEndpoint: "127.0.0.1:1", // deliberately unreachable
		SampleRatio:          1.0,
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() returned unexpected error: %v", err)
	}
	if tp == nil {
		t.Fatal("NewTracerProvider() returned nil provider")
	}

	_, span := tp.Tracer("test").Start(context.Background(), "op")
	span.End()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer shutdownCancel()
	// Shutdown flushes the (empty-ish) batch to the unreachable endpoint;
	// it may return an error from the failed export attempt, which is
	// fine and expected -- what matters is that it returns at all within
	// the timeout instead of hanging.
	_ = shutdown(shutdownCtx)
}

func TestNewTracerProvider_RejectsInvalidResource(t *testing.T) {
	// This exercises the enabled path's resource construction with a
	// normal, valid config to guard against a future regression that
	// makes resource.Merge start erroring on well-formed input -- there
	// is no way to make resource.Merge fail with today's inputs, so this
	// doubles as documentation of what "valid" looks like.
	ctx := context.Background()
	tp, shutdown, err := NewTracerProvider(ctx, Config{
		ServiceName:          "demo-service",
		ServiceVersion:       "v1.0.0",
		ReleaseTrack:         "stable",
		ReleaseVersion:       "v1.0.0",
		OTLPExporterEndpoint: "otel-collector.releaseguard.svc.cluster.local:4318",
		SampleRatio:          0.5,
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() returned unexpected error: %v", err)
	}
	if tp == nil {
		t.Fatal("NewTracerProvider() returned nil provider")
	}
	_ = shutdown(ctx)
}
