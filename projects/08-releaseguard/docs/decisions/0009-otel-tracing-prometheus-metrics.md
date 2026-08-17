# ADR 0009: OpenTelemetry for tracing, existing Prometheus client for metrics

**Status:** Accepted
**Decision ID:** Phase 2 scope clarification (not a spec §1.9 D-numbered decision)
**Date:** 2026-08-17

## Context

FR3 asks for "OpenTelemetry instrumentation + Prometheus metrics," and the
Phase 2 roadmap row names the deliverable "OTel traces/metrics, Prometheus
scrape/query." Read literally this could mean the demo-service should
adopt the OTel *Metrics* SDK (with its Prometheus exporter bridge) for
counters/histograms, not just OTel *tracing*.

Phase 0/1 already instrument the demo-service directly against
`github.com/prometheus/client_golang` (`internal/server/metrics.go`,
`middleware.go`): `http_requests_total`, `http_request_duration_seconds`
and `work_dependency_failures_total`, all labeled by route/method/status/
version/track, with 90%+ test coverage backing them
(`experiments/raw/phase0-test-coverage.md`). This is a working,
already-tested implementation of "Prometheus metrics... queryable per
stable/canary identity" (FR3), and Phase 2's own exit criterion is exactly
that queryability -- not "the metrics must originate from the OTel SDK
specifically."

## Decision

Split the two FR3 clauses across two different instrumentation paths:

- **Tracing** (genuinely new in Phase 2): the OTel Go SDK
  (`go.opentelemetry.io/otel/sdk/trace`), exporting spans via OTLP/HTTP to
  an in-cluster OpenTelemetry Collector (`observability/otel/`). Every
  instrumented route gets a request-level span (`internal/server/
  middleware.go`'s `instrument`), and `/work` opens a child span around
  the simulated dependency call (`internal/server/handlers.go`) --
  `release.track`/`release.version` are attached as resource attributes on
  every span a process emits, and per-request `http.route`/`http.method`/
  `http.status_code` attributes let the collector's logs answer "what did
  this specific canary request actually do" in a way a metric alone
  cannot.
- **Metrics** (unchanged from Phase 0/1): kept on `github.com/prometheus/
  client_golang`, scraped directly by the Prometheus server this phase
  adds (`observability/prometheus/`), with `version`/`track` remaining
  first-class Prometheus labels the way FR3 asks.

## Consequences

- No churn on already-tested, working metrics code or the tests asserting
  on exact metric/label names (`TestMetricsEndpoint_ExposesKnownMetricNames`,
  `TestInstrument_Records*`) -- rewriting them onto the OTel Metrics SDK's
  Prometheus exporter would risk subtly different exposition-format output
  (unit-suffix normalization, bucket boundary handling) for zero
  behavioral gain, since the end state ("Prometheus-format metrics on
  `/metrics`, labeled by track/version") is identical either way.
  demo-service must stay `go test -race` clean throughout this project;
  this keeps that guarantee cheap to hold.
- Two telemetry libraries (`client_golang` for metrics, the OTel SDK for
  traces) inside one small service is slightly more surface area than one
  library doing both, but it is exactly what a real-world Go service
  commonly looks like today (the OTel *Metrics* SDK is comparatively young
  and Prometheus's own client library remains extremely common even in
  OTel-tracing shops) -- an interview-defensible, not merely convenient,
  choice.
- The evaluator (Phase 4+) queries Prometheus for every SLO signal
  (ADR 0004); trace data is diagnostic evidence for a human investigating
  *why* a decision was made, not an input to the decision itself. Keeping
  tracing and metrics on separate pipelines matches that separation of
  concerns instead of forcing one pipeline to serve both purposes.

## Alternatives considered

- **OTel Metrics SDK + Prometheus exporter bridge for everything**: the
  most literal reading of "OpenTelemetry instrumentation," and the
  textbook-correct architecture for a greenfield service. Rejected for
  *this* codebase specifically because Phase 0/1 already shipped and
  tested a working Prometheus-native metrics path before Phase 2 existed;
  migrating it now is pure churn risk against the "no wholesale rewrite"
  and "stay `go test -race` clean" constraints, for an outcome
  indistinguishable from outside the service (the `/metrics` output looks
  the same either way). A greenfield service without that history would
  reasonably choose the unified OTel path instead.
- **OTel traces routed through the Collector's `spanmetrics` connector**
  to also derive RED metrics from spans: interesting and would technically
  satisfy "OTel traces/metrics" from one pipeline, but adds a second,
  redundant source of the same request-rate/error-rate/latency numbers
  the existing Prometheus metrics already provide, plus extra Collector
  configuration complexity, for a demo service whose whole point is to be
  "deliberately simple" (spec §1.2). Not pursued in the MVP.
