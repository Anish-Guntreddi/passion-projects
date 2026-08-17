# ADR 0007: Absolute thresholds first, relative guardrails second

**Status:** Accepted (implementation lands in Phase 4, not this change)
**Decision ID:** D6 (spec §1.9)
**Date:** 2026-08-17

## Context

D6 asks whether the MVP policy engine should evaluate absolute SLO
thresholds (e.g. "P95 latency < 300ms"), relative canary-vs-baseline
comparisons (e.g. "canary P95 latency < 1.2x stable P95 latency"), or
both, with a recommended default of "absolute thresholds first, relative
guardrails second."

## Decision

Phase 4's policy schema supports absolute thresholds as the primary,
required signal type (FR4: "absolute SLO thresholds"), and adds relative
canary-vs-baseline guardrails as an optional, additive layer on top ("
relative canary-vs-baseline guardrails where useful," FR4) -- not the
other way around. A policy that only defines absolute thresholds must be
valid and sufficient to drive promote/rollback decisions; relative
guardrails, when present, add an additional pass/fail condition rather
than replacing the absolute check.

## Consequences

- Absolute thresholds are simpler to reason about and to demo (the
  failure-injection scenarios in §1.8 -- latency regression, error
  regression -- are naturally described as "P95 exceeded Xms" first, with
  "and it's Y times worse than stable" as a secondary, more nuanced
  signal). This ordering keeps Phase 4's initial unit-test surface
  smaller and its exit criterion ("recorded telemetry fixtures produce
  deterministic decisions") reachable without first solving relative
  baseline-tracking.
- Relative guardrails require a live "stable" baseline to compare against,
  which introduces its own missing-data question (what if stable's own
  telemetry is sparse?). Building absolute thresholds first means that
  question is isolated to the relative-guardrail code path, not tangled
  into the base case.
- Threshold *values* are explicitly out of scope for this decision and
  for automated implementation: per spec §1.9, "policy threshold values
  are human-review decisions." ADR 0007 fixes the *shape* of the policy
  (absolute-first, relative-additive); it does not choose numbers. Phase 4
  will flag concrete threshold values for human review before they land,
  per the Claude Code handoff's per-task requirements.

## Alternatives considered

- **Relative-only** (canary vs. stable comparison as the sole signal):
  rejected -- fails Kubernetes handoff rule 1 (never auto-promote with
  missing evidence) in the specific case where *stable itself* is
  unhealthy: a relative-only policy could let an equally-bad canary
  "pass" by comparison. Absolute thresholds are the safety floor that
  relative guardrails sit on top of, not a replacement for them.
- **Absolute-only, permanently**: leaves out a genuinely useful signal
  (canaries that are absolutely within SLO but meaningfully worse than
  the currently-running stable version) explicitly called out by FR4;
  deferred to "second," not dropped.
