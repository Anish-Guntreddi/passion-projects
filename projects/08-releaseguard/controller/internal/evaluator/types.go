// Package evaluator implements Phase 4's single-window evaluation: given a
// policy and a metrics.Querier, decide whether one observation window for
// a track is HEALTHY, DEGRADED or INCONCLUSIVE, with the full evidence
// trail (per-signal values, thresholds, reason codes) needed to answer
// "why did ReleaseGuard make this decision?" (FR4). It does not decide
// PROMOTE/PAUSE/ROLLBACK or track consecutive windows across time -- that
// is internal/rollout's job (Phase 5), built on top of a sequence of
// WindowResults this package produces.
package evaluator

import "time"

// Classification is a single window's health verdict. It intentionally
// mirrors the HEALTHY/DEGRADED/INCONCLUSIVE sub-states in the spec's
// decision state machine (FR4): internal/rollout.Rollout enters exactly
// one of the correspondingly-named States per window, driven by this
// value.
type Classification string

const (
	Healthy      Classification = "HEALTHY"
	Degraded     Classification = "DEGRADED"
	Inconclusive Classification = "INCONCLUSIVE"
)

// Reason codes explaining a Classification or a single SignalResult.
// Stable, machine-checkable strings (not free-form prose) so a human or a
// script reading the audit log (ADR 0005) can filter/aggregate on them.
const (
	ReasonAllSignalsHealthy   = "ALL_SIGNALS_WITHIN_THRESHOLD"
	ReasonThresholdBreached   = "ABSOLUTE_THRESHOLD_BREACHED"
	ReasonRelativeBreached    = "RELATIVE_GUARDRAIL_BREACHED"
	ReasonMissingTelemetry    = "MISSING_TELEMETRY"
	ReasonInsufficientSamples = "INSUFFICIENT_SAMPLES"
)

// SignalResult is one signal's evidence for one window: what was queried,
// what came back, and whether it passed. Missing is true when the
// underlying query returned metrics.ErrNoData (for either the canary value
// or, when a relative guardrail is configured, the baseline value) --
// Value/BaselineValue are meaningless when Missing is true and must not be
// read.
type SignalResult struct {
	Name     string `json:"name"`
	Optional bool   `json:"optional"`

	Query    string  `json:"query"`
	Value    float64 `json:"value,omitempty"`
	HasValue bool    `json:"has_value"`

	BaselineQuery string  `json:"baseline_query,omitempty"`
	BaselineValue float64 `json:"baseline_value,omitempty"`
	HasBaseline   bool    `json:"has_baseline,omitempty"`

	ThresholdPass bool  `json:"threshold_pass"`
	RelativePass  *bool `json:"relative_pass,omitempty"`

	Missing     bool     `json:"missing"`
	ReasonCodes []string `json:"reason_codes,omitempty"`
}

// Healthy reports whether this signal, on its own, is within bounds:
// present, passing its absolute threshold, and (if configured) passing its
// relative guardrail. A Missing signal is never Healthy -- see Claude Code
// handoff rule 1.
func (r SignalResult) Healthy() bool {
	if r.Missing {
		return false
	}
	if !r.ThresholdPass {
		return false
	}
	if r.RelativePass != nil && !*r.RelativePass {
		return false
	}
	return true
}

// WindowResult is one evaluated observation window for one track: the
// Classification plus every SignalResult that produced it. This is the
// unit internal/rollout consumes one at a time to drive the state machine,
// and the unit internal/audit persists as evidence.
type WindowResult struct {
	Timestamp      time.Time      `json:"timestamp"`
	Track          string         `json:"track"`
	Classification Classification `json:"classification"`

	// SampleCount is the measured traffic volume for the sample-count
	// gate, nil if the policy didn't configure one (MinSampleCount == 0).
	SampleCount *float64 `json:"sample_count,omitempty"`

	SignalResults []SignalResult `json:"signal_results,omitempty"`
	ReasonCodes   []string       `json:"reason_codes,omitempty"`
}
