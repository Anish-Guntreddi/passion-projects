# ADR 0004: Prometheus access via instant + range HTTP queries behind an interface

**Status:** Accepted (implementation lands in Phase 2/4, not this change)
**Decision ID:** D3 (spec §1.9)
**Date:** 2026-08-17

## Context

D3 asks how the controller should query telemetry, with a recommended
default of "instant + range queries via HTTP API behind a metrics-query
interface." Recorded now, ahead of Phase 2 (Telemetry) and Phase 4
(evaluator), per the spec's ADR-before-phase guidance.

## Decision

The controller queries Prometheus's HTTP API directly:

- **Instant queries** (`/api/v1/query`) for "what is the current value of
  this signal" checks (e.g. current error rate over the last evaluation
  window, expressed as a PromQL range-vector-to-scalar expression such as
  `rate(http_requests_total{status=~"5..",track="canary"}[5m])`).
- **Range queries** (`/api/v1/query_range`) for window-based aggregation
  and for building the evidence attached to audit records (e.g. "show me
  P95 latency for the canary over the observation window" as a time
  series, not just a single number, so the audit log can explain *why* a
  decision was made with a graphable trace, not just a verdict).

Both go through a `MetricsQuerier` interface (name indicative; finalized
in Phase 4) that the evaluator depends on, rather than the evaluator
importing a Prometheus client directly. This is what makes the evaluator's
Phase 4 unit tests possible without a running Prometheus: tests provide a
fake `MetricsQuerier` backed by fixture data, while the real
implementation wraps `github.com/prometheus/client_golang/api/prometheus/v1`
against a live server (wired up in Phase 2, exercised end-to-end from
Phase 6 onward).

## Consequences

- The evaluator's core decision logic (threshold comparisons, window
  aggregation, state transitions) is testable in pure Go with fixture
  data and zero network/cluster dependency -- directly enabling the spec's
  Phase 4 exit criterion ("recorded telemetry fixtures produce
  deterministic decisions") and §1.7's unit-test requirements.
- A missing/unreachable Prometheus, or a query that returns no samples,
  surfaces as an explicit error/empty-result from `MetricsQuerier`, which
  the evaluator must treat as "insufficient evidence" (Claude Code handoff
  rule 1: never auto-promote with missing evidence) rather than a Go
  panic or a silently-interpreted zero.
- Two query shapes (instant + range) rather than one keeps the interface
  slightly larger, but avoids forcing every window-aggregation call
  through a range query when a single instant PromQL expression already
  does the aggregation server-side (e.g. `rate(...)[5m]`), which is both
  faster and simpler for the common case.

## Alternatives considered

- **OTel Collector metrics pipeline queried directly** (skip Prometheus,
  query the collector or an OTLP-native backend): rejected because
  Prometheus's PromQL + HTTP API is the de facto standard for exactly this
  "query current/recent SLO signals" use case, and the tech-stack table
  already commits to Prometheus as the query surface, with OTel as the
  instrumentation/export path feeding it.
- **Push-based evaluator** (demo-service pushes decision-relevant
  aggregates to the controller directly, bypassing Prometheus): rejected
  because it would require designing and versioning a custom push
  protocol and would defeat the point of using industry-standard
  observability tooling as a portfolio signal.
