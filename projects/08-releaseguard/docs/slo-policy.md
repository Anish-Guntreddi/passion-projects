# ReleaseGuard policy schema (Phase 4)

This describes the typed policy config FR4 requires -- what a policy file
says, how the evaluator uses it, and which values in
[`controller/policy/examples/demo-service-default.yaml`](../controller/policy/examples/demo-service-default.yaml)
are placeholders still needing human review versus structural defaults
already justified by an ADR. See
[`controller/internal/policy`](../controller/internal/policy) for the Go
types and validation this document describes, and
[`docs/architecture.md`](architecture.md#controller-phases-4-5) for how a
policy fits into the evaluator and state machine around it.

## Why one schema for YAML and JSON

`controller/internal/policy` uses `sigs.k8s.io/yaml`, which converts YAML
to JSON and then decodes via `encoding/json`. One set of `json:` struct
tags therefore drives both formats -- a `.yaml` and an equivalent `.json`
policy document parse into an identical `Policy` struct with no separate
tag set to keep in sync (`controller/internal/policy/testdata/valid.yaml`
and `valid.json` are two independently-loaded fixtures proving this).
Unknown fields are rejected (`yaml.UnmarshalStrict`) so a typo in a policy
file fails loudly at load time instead of silently taking its zero value.

## Shape

```yaml
name: demo-service-canary
version: 1

evaluation:
  interval: 15s                    # how often a live reconcile loop re-evaluates (Phase 6)
  window: 30s                      # PromQL lookback per evaluation, substituted as {{.Window}}
  min_sample_count: 20             # minimum traffic (FR4) before a verdict is trusted
  sample_count_query: sum(increase(http_requests_total{track="{{.Track}}"}[{{.Window}}]))
  consecutive_healthy_windows: 3   # windows required before PROMOTE
  consecutive_unhealthy_windows: 2 # windows required before ROLLBACK, and independently
                                    # before the on_missing_telemetry action fires
  max_rollout_duration: 30m        # deadline; never resolves to PROMOTE (see below)
  on_missing_telemetry: pause      # "pause" | "rollback" -- no "promote" value exists
  baseline_track: stable           # what a relative guardrail compares the canary against

signals:
  - name: error_rate
    query: sum(rate(http_requests_total{status=~"5..",track="{{.Track}}"}[{{.Window}}])) / ...
    threshold:
      max: 0.02
    relative:
      max_ratio: 3.0
    optional: false
```

Every signal's `query` is a Go `text/template` string, rendered once per
track evaluated (`{{.Track}}` -> `canary` always, and again for
`baseline_track` when `relative` is set) with `{{.Window}}` substituted
from `evaluation.window`. `policy.Validate` requires every query
(`signals[].query` and `evaluation.sample_count_query`) to actually
contain `{{.Track}}` -- a query that doesn't vary by track can't measure
"is *this* track healthy," and for a `relative` guardrail specifically, a
track-invariant query would render identically for canary and baseline,
making the comparison trivially 1:1 regardless of real values. Both are
config-authoring bugs caught at load time, not evaluation-time surprises.

## Absolute first, relative additive (ADR 0007)

`threshold.max`/`threshold.min` are the required, primary check. `relative`
is optional and, per [ADR 0007](decisions/0007-absolute-vs-relative-thresholds.md),
**additive**: a signal with both configured must pass *both* to count as
healthy. A canary that is absolutely fine but far worse than a healthy
stable is still flagged; a canary that's absolutely broken doesn't get a
pass just because stable happens to be broken the same way (see ADR 0007's
"Alternatives considered" for why relative-only was rejected).

`evalRelative` (`controller/internal/evaluator/evaluator.go`) special-cases
a zero (or negative) baseline explicitly rather than dividing by it: a zero
baseline with a nonzero canary fails a `max_ratio` bound (an
undefined-but-clearly-large ratio should not silently pass); a zero
baseline with a zero canary is the degenerate 0/0 case and is treated as
within bounds. Both are covered by
`TestEvalRelative_ZeroBaselineEdgeCases`.

## Missing telemetry never means healthy (rule 1)

Three independent things happen when evidence is missing or insufficient,
matching Claude Code handoff rule 1 ("never auto-promote with missing
evidence") at every layer rather than one:

1. **Evaluator**: a signal whose query returns `metrics.ErrNoData` (no
   series, or Prometheus's NaN for a 0/0 division) is `Missing`, and a
   `Missing` *required* signal makes the whole window `INCONCLUSIVE` --
   never `HEALTHY` -- regardless of how every other signal looks
   (`TestEvaluate_InconclusiveOnMissingSignal_NeverHealthy`). A signal
   marked `optional: true` is the one exception: its absence alone does
   not force `INCONCLUSIVE` (FR4's example: "optional business-success
   metric"), though a *present but failing* optional signal still
   degrades the window like any other.
2. **Sample-count gate**: below `min_sample_count`, or the count query
   itself returning no data, is `INCONCLUSIVE` before any signal is even
   queried -- there's no point checking thresholds against a window that
   doesn't have enough evidence to trust in the first place.
3. **Relative guardrail baseline**: if the *baseline* track's value for a
   `relative`-guarded signal is missing, the signal is reported missing --
   not silently passed, not silently failed -- **provided the canary's own
   absolute threshold hasn't already independently failed**
   (`TestEvaluate_MissingBaselineMakesRelativeSignalMissing`). If the
   canary's absolute threshold *has* already failed, that verdict stands
   on its own (ADR 0007: absolute thresholds first) rather than being
   discarded as "missing" just because the relative guardrail couldn't
   also be checked -- a missing baseline must never downgrade a known
   breach into a falsely lenient `INCONCLUSIVE`
   (`TestEvaluate_MissingBaselineDoesNotMaskAbsoluteBreach`). The missing
   baseline is still recorded in that signal's `reason_codes` alongside
   the threshold breach, so the audit trail shows the relative guardrail
   was never evaluated that window.

`on_missing_telemetry` (`pause` or `rollback`, default `pause`) governs
what the rollout *does* once `INCONCLUSIVE` windows persist for
`consecutive_unhealthy_windows` in a row -- see
[ADR 0010](decisions/0010-missing-telemetry-and-pause-semantics.md) for why
`pause` is the default and what "paused" actually resolves to today.
`max_rollout_duration` expiring is handled the same way, independent of
`on_missing_telemetry`: it always pauses with decision `INCONCLUSIVE`,
because a deadline is an absence of a conclusive decision, never evidence
of health.

## Flagged for human review

Per spec §1.9 ("policy threshold values are human-review decisions") and
the kickoff prompt ("flag all policy threshold values for my review before
Phase 4 lands"): every numeric value in
`controller/policy/examples/demo-service-default.yaml` --
`window`/`interval`/`min_sample_count`/the consecutive-window counts/
`max_rollout_duration`, and every signal's `threshold`/`relative` bound --
is a placeholder derived from demo-service's own configured defaults
(`BASE_LATENCY_MS=20`, `LATENCY_JITTER_MS=10`, `ERROR_RATE=0` --
see the top-level README's demo-service configuration table), **not** a
measured SLO agreed with a real stakeholder. The file itself repeats this
flag inline next to every value. `on_missing_telemetry`'s default
(`pause`) is a structural safety default justified by ADR 0010, not a
threshold value, and does not need the same review, though a specific
policy choosing `rollback` instead is exactly the kind of per-policy call
a human should make deliberately.

## Why `dependency_failure_rate` stands in for "saturation/resource signal"

FR4's example signal list includes "saturation/resource signal." demo-
service does not currently export a CPU/memory gauge over `/metrics` (see
`demo-service/internal/server/metrics.go`) -- `MEMORY_PRESSURE_MB` is a
fault-injection *input* (an env var), not an exported *metric*. Rather than
add unexercised instrumentation to demo-service as a Phase 4/5 side
effect (out of this phase's scope, and Phase 7's failure-injection work is
the natural place to decide what a real saturation signal should look
like), the default policy's fourth signal, `dependency_failure_rate`, uses
`work_dependency_failures_total` -- a real, already-instrumented signal
distinct from `error_rate` (HTTP 5xx) -- as a present-day stand-in for
"something beyond latency/errors can independently degrade a release."
Revisit alongside Phase 7 once a real resource/saturation metric exists.

## Local reproducibility

```bash
cd controller
go run ./cmd/controller validate -policy policy/examples/demo-service-default.yaml
go run ./cmd/controller simulate \
    -policy policy/examples/demo-service-default.yaml \
    -scenario scenarios/a-healthy-promotion.json
```

See [`docs/architecture.md`](architecture.md#controller-phases-4-5) for
what `simulate` actually runs (the same `internal/scenario.Run` the Phase
4/5 exit-criterion tests call), and
[`experiments/raw/phase4-5-scenario-evidence.md`](../experiments/raw/phase4-5-scenario-evidence.md)
for a captured, real run of all five spec §1.8 scenarios against this
policy file.
