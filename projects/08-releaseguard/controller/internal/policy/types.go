// Package policy defines ReleaseGuard's typed rollout-evaluation policy
// schema (spec FR4: "Typed policy config (YAML/JSON with validation)") and
// the query-templating helper the evaluator (internal/evaluator) uses to
// turn a signal's query template into a concrete, track-scoped PromQL
// string.
//
// A Policy is parsed once (Load/Parse), defaulted (ApplyDefaults) and
// validated (Validate) before it is handed to an evaluator -- an invalid
// or under-specified policy is a config error caught at load time, not a
// runtime surprise during a live rollout.
package policy

// Policy is the top-level typed rollout-evaluation policy document. It is
// deliberately independent of any specific Kubernetes object or Prometheus
// server: it says *what* to check and *how strict* to be, not *how* to
// reach the metrics backend (that's internal/metrics.Querier) or *what to
// do* with a verdict (that's internal/rollout).
type Policy struct {
	// Name identifies the policy for audit records and CLI output (e.g.
	// "demo-service-canary"). Required.
	Name string `json:"name"`

	// Version is the policy schema version. Only 1 is currently accepted;
	// present from day one so a future breaking schema change has
	// somewhere to hang a migration off of instead of silently
	// misinterpreting an old document.
	Version int `json:"version"`

	// Evaluation controls window sizing, sample-count gating, consecutive-
	// window counts, the rollout deadline and the missing-telemetry
	// action -- everything FR4 lists that is not itself a per-signal
	// threshold.
	Evaluation EvaluationConfig `json:"evaluation"`

	// Signals is the list of metrics/signals evaluated every window (FR4:
	// "metrics/signals"). At least one is required.
	Signals []SignalConfig `json:"signals"`
}

// MissingTelemetryAction is the policy-configurable action taken when a
// window's evidence is missing or insufficient (FR4: "action on missing
// telemetry"). Deliberately a closed enum with no "promote"-equivalent
// value -- Claude Code handoff rule 1 ("never auto-promote with missing
// evidence") is enforced structurally here, not just by convention: there
// is no value an operator could configure that would make missing
// telemetry resolve to a promotion.
type MissingTelemetryAction string

const (
	// OnMissingPause pauses the rollout (decision PAUSE) when telemetry
	// stays missing/insufficient for the configured consecutive-window
	// count. The safer of the two options -- it stops and waits for a
	// human rather than taking any action on absent evidence -- and is
	// therefore the default (see ApplyDefaults).
	OnMissingPause MissingTelemetryAction = "pause"

	// OnMissingRollback rolls the canary back (decision ROLLBACK) under
	// the same condition. An operator can choose this for a policy where
	// "we can't observe it" should be treated as failure-equivalent
	// rather than merely inconclusive.
	OnMissingRollback MissingTelemetryAction = "rollback"
)

// EvaluationConfig groups every evaluation-cadence and safety-guardrail
// setting FR4 requires that is not itself a signal.
type EvaluationConfig struct {
	// Interval is how often a live evaluation loop (Phase 6) re-runs a
	// window evaluation. Not consumed by the evaluator/state-machine logic
	// itself (which is driven by discrete window results, real-time or
	// simulated) -- it is policy metadata for the future reconcile loop,
	// validated here so a bad value is caught at policy-load time rather
	// than Phase 6.
	Interval Duration `json:"interval"`

	// Window is the PromQL lookback duration substituted into every
	// signal's query template as {{.Window}} (e.g. "30s" -> `[30s]`
	// inside a `rate(...)` call).
	Window Duration `json:"window"`

	// MinSampleCount is the minimum traffic volume (FR4: "minimum sample
	// count/traffic") a window must have observed, per SampleCountQuery,
	// before its verdict is trusted. Zero disables the sample-count gate
	// (per-signal missing-data detection still applies independently).
	MinSampleCount int `json:"min_sample_count"`

	// SampleCountQuery is the query template (same {{.Track}}/{{.Window}}
	// templating as a signal's Query) the evaluator renders and executes
	// to measure traffic volume for the sample-count gate. Required when
	// MinSampleCount > 0.
	SampleCountQuery string `json:"sample_count_query,omitempty"`

	// ConsecutiveHealthyWindows is how many consecutive HEALTHY windows
	// are required before the rollout decides PROMOTE.
	ConsecutiveHealthyWindows int `json:"consecutive_healthy_windows"`

	// ConsecutiveUnhealthyWindows is how many consecutive DEGRADED windows
	// are required before the rollout decides ROLLBACK, and independently
	// how many consecutive INCONCLUSIVE windows are required before
	// OnMissingTelemetry's action fires. Reusing one threshold for both
	// keeps the schema small; nothing prevents a future schema version
	// from splitting it if a real policy needs different tolerances for
	// "actively bad" versus "can't tell."
	ConsecutiveUnhealthyWindows int `json:"consecutive_unhealthy_windows"`

	// MaxRolloutDuration is the deadline (FR4: "maximum rollout duration")
	// measured from the moment observation begins. If it elapses without
	// a PROMOTE or ROLLBACK decision, the rollout is paused with decision
	// INCONCLUSIVE -- never promoted; a deadline is not evidence of
	// health.
	MaxRolloutDuration Duration `json:"max_rollout_duration"`

	// OnMissingTelemetry is the action taken when telemetry stays
	// missing/insufficient for ConsecutiveUnhealthyWindows windows in a
	// row. Defaults to OnMissingPause (see ApplyDefaults) if left unset.
	OnMissingTelemetry MissingTelemetryAction `json:"on_missing_telemetry,omitempty"`

	// BaselineTrack is the release track relative guardrails compare the
	// canary against. Defaults to "stable" (see ApplyDefaults), matching
	// the track naming convention used everywhere else in this repo
	// (demo-service RELEASE_TRACK, ADR 0003's replica-count split).
	BaselineTrack string `json:"baseline_track,omitempty"`
}

// Threshold is an absolute SLO threshold (FR4: "absolute SLO thresholds";
// ADR 0007: absolute thresholds are the required, primary signal type). At
// least one of Max/Min must be set. Pointers distinguish "not configured"
// from "configured as zero" (e.g. a signal legitimately thresholded at
// exactly 0).
type Threshold struct {
	Max *float64 `json:"max,omitempty"`
	Min *float64 `json:"min,omitempty"`
}

// RelativeGuardrail is an optional canary-vs-baseline comparison (FR4:
// "relative canary-vs-baseline guardrails where useful"; ADR 0007:
// additive on top of the absolute threshold, never a replacement for it).
// At least one of MaxRatio/MinRatio must be set.
type RelativeGuardrail struct {
	// MaxRatio: canary/baseline must not exceed this ratio.
	MaxRatio *float64 `json:"max_ratio,omitempty"`
	// MinRatio: canary/baseline must not fall below this ratio.
	MinRatio *float64 `json:"min_ratio,omitempty"`
}

// SignalConfig is one evaluated metric/signal (FR4: "metrics/signals").
type SignalConfig struct {
	// Name identifies the signal in reason codes, audit evidence and CLI
	// output (e.g. "error_rate"). Required, unique within a policy.
	Name string `json:"name"`

	// Query is a PromQL query *template* containing the literal
	// placeholders {{.Track}} and (optionally) {{.Window}}, rendered by
	// RenderQuery once per track evaluated (canary always; baseline too
	// when Relative is set). Required; must contain {{.Track}} (see
	// Validate) so a canary-scoped and baseline-scoped render can never
	// collapse to the same query.
	Query string `json:"query"`

	// Threshold is this signal's absolute pass/fail bound.
	Threshold Threshold `json:"threshold"`

	// Relative is this signal's optional canary-vs-baseline guardrail.
	Relative *RelativeGuardrail `json:"relative,omitempty"`

	// Optional marks a signal whose absence (missing telemetry) does not,
	// by itself, make the whole window INCONCLUSIVE -- for genuinely
	// optional signals (FR4's example: "optional business-success
	// metric"). A present-but-failing optional signal still degrades the
	// window; only *missing* data is treated leniently, and only for this
	// signal specifically.
	Optional bool `json:"optional,omitempty"`
}
