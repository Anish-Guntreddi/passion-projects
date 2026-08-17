// Package telemetry builds the demo-service's OpenTelemetry TracerProvider.
//
// This is deliberately the *only* thing this package does. Prometheus
// metrics remain instrumented directly against github.com/prometheus/
// client_golang (see internal/server/metrics.go) rather than being
// rebuilt on top of the OTel Metrics SDK's Prometheus exporter -- see
// docs/decisions/0009-otel-tracing-prometheus-metrics.md for why that
// split is deliberate rather than an oversight.
package telemetry

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// Config controls how the demo-service exports trace spans.
type Config struct {
	// ServiceName is the OTel "service.name" resource attribute (e.g.
	// "releaseguard-demo-service"). Constant across every instance of the
	// demo-service regardless of track/version.
	ServiceName string

	// ServiceVersion is the OTel "service.version" resource attribute --
	// the image version baked in at build time (internal/version.Current().
	// Version), identifying *what code is running*.
	ServiceVersion string

	// ReleaseTrack and ReleaseVersion mirror config.Config's fields of the
	// same name and are attached as resource attributes so every span
	// emitted by this process is identifiable as stable/canary and which
	// rollout it represents without having to repeat them on every span.
	ReleaseTrack   string
	ReleaseVersion string

	// OTLPExporterEndpoint is the collector's host:port for the OTLP/HTTP
	// exporter. Empty disables export -- NewTracerProvider returns a
	// genuine no-op provider in that case rather than one pointed at a
	// dead address, so local development and unit tests never attempt a
	// network call.
	OTLPExporterEndpoint string

	// SampleRatio is the fraction (in [0,1]) of traces sampled, applied
	// via ParentBased(TraceIDRatioBased(SampleRatio)) so an incoming
	// traceparent header's sampling decision is still honored when
	// present.
	SampleRatio float64
}

// ShutdownFunc flushes any buffered spans and releases exporter resources.
// It is always safe to call exactly once, and is a no-op when tracing was
// never enabled (empty OTLPExporterEndpoint).
type ShutdownFunc func(context.Context) error

// NewTracerProvider builds a trace.TracerProvider for cfg.
//
// It returns the trace.TracerProvider *interface* (not the concrete
// *sdktrace.TracerProvider* type) deliberately: every caller in this
// codebase (internal/server.Server) only ever needs .Tracer(name), which
// is also exactly what a no-op or test provider implements -- narrowing
// the return type keeps callers from accidentally depending on SDK-only
// methods that a test double would not have.
func NewTracerProvider(ctx context.Context, cfg Config) (trace.TracerProvider, ShutdownFunc, error) {
	if cfg.OTLPExporterEndpoint == "" {
		return noop.NewTracerProvider(), func(context.Context) error { return nil }, nil
	}

	client := otlptracehttp.NewClient(
		otlptracehttp.WithEndpoint(cfg.OTLPExporterEndpoint),
		// The demo-service talks to an in-cluster collector Service over
		// the pod network -- there is no TLS termination in front of it
		// for this MVP (see docs/decisions/0009), so plaintext is correct
		// here rather than a workaround.
		otlptracehttp.WithInsecure(),
	)
	exporter, err := otlptrace.New(ctx, client)
	if err != nil {
		return nil, nil, fmt.Errorf("telemetry: creating OTLP trace exporter: %w", err)
	}

	res, err := resource.Merge(
		resource.Default(),
		resource.NewSchemaless(
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion(cfg.ServiceVersion),
			attribute.String("release.track", cfg.ReleaseTrack),
			attribute.String("release.version", cfg.ReleaseVersion),
		),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("telemetry: building resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(cfg.SampleRatio))),
	)

	return tp, tp.Shutdown, nil
}
