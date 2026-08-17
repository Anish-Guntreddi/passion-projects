package evaluator

import (
	"context"
	"errors"
	"fmt"
	"time"

	"releaseguard/controller/internal/metrics"
	"releaseguard/controller/internal/policy"
)

// Evaluator runs one policy's signals against one metrics.Querier for a
// given track, producing a WindowResult per call to Evaluate. It holds no
// state across windows -- consecutive-window tracking and the
// promote/pause/rollback decision live in internal/rollout, which calls
// Evaluate once per window.
type Evaluator struct {
	Policy  *policy.Policy
	Querier metrics.Querier
}

// New builds an Evaluator. p and q must be non-nil.
func New(p *policy.Policy, q metrics.Querier) *Evaluator {
	return &Evaluator{Policy: p, Querier: q}
}

// Evaluate runs every configured signal (and, if configured, the
// sample-count gate) for track at instant now, and classifies the window.
//
// The returned error is non-nil only for a Querier failure that is not
// "no data" (e.g. Prometheus itself is unreachable, §1.7's "Prometheus
// unavailable" failure scenario) -- a well-formed but empty/NaN result is
// not an error, it is evidence, and is reported as Classification
// INCONCLUSIVE with reason MISSING_TELEMETRY/INSUFFICIENT_SAMPLES so the
// caller's audit trail explains *why*, rather than the caller having to
// distinguish "the evaluator crashed" from "the evaluator correctly
// refused to decide."
func (e *Evaluator) Evaluate(ctx context.Context, track string, now time.Time) (WindowResult, error) {
	windowStr := e.Policy.Evaluation.Window.Duration.String()
	result := WindowResult{Timestamp: now, Track: track}

	if e.Policy.Evaluation.MinSampleCount > 0 {
		inconclusive, err := e.checkSampleCount(ctx, track, windowStr, now, &result)
		if err != nil {
			return WindowResult{}, err
		}
		if inconclusive {
			return result, nil
		}
	}

	// allHealthy tracks every signal that isn't given the optional-missing
	// pass below -- that includes present-but-failing optional signals,
	// per policy.SignalConfig.Optional's doc comment ("A present-but-
	// failing optional signal still degrades the window; only *missing*
	// data is treated leniently, and only for this signal specifically").
	allHealthy := true
	anyRequiredMissing := false

	for _, sig := range e.Policy.Signals {
		sr, err := e.evaluateSignal(ctx, sig, track, windowStr, now)
		if err != nil {
			return WindowResult{}, fmt.Errorf("evaluator: signal %q: %w", sig.Name, err)
		}
		result.SignalResults = append(result.SignalResults, sr)

		if sig.Optional && sr.Missing {
			// A missing optional signal is the one case treated leniently:
			// it must not force the window INCONCLUSIVE, nor block HEALTHY.
			continue
		}
		if sr.Missing {
			anyRequiredMissing = true
			continue
		}
		if !sr.Healthy() {
			allHealthy = false
		}
	}

	switch {
	case anyRequiredMissing:
		result.Classification = Inconclusive
		result.ReasonCodes = []string{ReasonMissingTelemetry}
	case !allHealthy:
		result.Classification = Degraded
		result.ReasonCodes = degradedReasonCodes(result.SignalResults)
	default:
		result.Classification = Healthy
		result.ReasonCodes = []string{ReasonAllSignalsHealthy}
	}
	return result, nil
}

// degradedReasonCodes derives the window-level reason codes for a DEGRADED
// classification from the actual per-signal verdicts, rather than
// hardcoding one reason for every degraded window -- FR4's audit trail
// must say *why* ReleaseGuard degraded a canary (absolute breach, relative
// breach, or both), not just that it did. A signal that is Missing or
// already Healthy contributes nothing (Missing signals never reach here
// for a DEGRADED window: a missing *required* signal forces INCONCLUSIVE
// instead, and a missing *optional* signal is skipped entirely above).
func degradedReasonCodes(results []SignalResult) []string {
	seen := make(map[string]bool, 2)
	var reasons []string
	add := func(code string) {
		if !seen[code] {
			seen[code] = true
			reasons = append(reasons, code)
		}
	}
	for _, sr := range results {
		if sr.Missing || sr.Healthy() {
			continue
		}
		if !sr.ThresholdPass {
			add(ReasonThresholdBreached)
		}
		if sr.RelativePass != nil && !*sr.RelativePass {
			add(ReasonRelativeBreached)
		}
	}
	if len(reasons) == 0 {
		// Should be unreachable (allHealthy is only false because some
		// signal is unhealthy, which always sets one of the reasons
		// above), but fail safe rather than hand back an empty evidence
		// trail for a classification that claims a breach happened.
		reasons = []string{ReasonThresholdBreached}
	}
	return reasons
}

// checkSampleCount runs the sample-count gate. It returns inconclusive =
// true (with result already populated as an INCONCLUSIVE verdict) when
// telemetry is missing or traffic is below the configured minimum, in
// which case the caller must stop and return result as-is without
// evaluating any signals -- there is no point checking thresholds against
// a window that doesn't have enough evidence to trust in the first place.
func (e *Evaluator) checkSampleCount(ctx context.Context, track, windowStr string, now time.Time, result *WindowResult) (inconclusive bool, err error) {
	q, err := policy.RenderQuery(e.Policy.Evaluation.SampleCountQuery, policy.TemplateContext{Track: track, Window: windowStr})
	if err != nil {
		return false, fmt.Errorf("evaluator: render sample_count_query: %w", err)
	}
	sample, err := e.Querier.InstantQuery(ctx, q, now)
	switch {
	case errors.Is(err, metrics.ErrNoData):
		result.Classification = Inconclusive
		result.ReasonCodes = []string{ReasonMissingTelemetry}
		return true, nil
	case err != nil:
		return false, fmt.Errorf("evaluator: sample_count_query: %w", err)
	}

	v := sample.Value
	result.SampleCount = &v
	if v < float64(e.Policy.Evaluation.MinSampleCount) {
		result.Classification = Inconclusive
		result.ReasonCodes = []string{ReasonInsufficientSamples}
		return true, nil
	}
	return false, nil
}

// evaluateSignal renders and executes one signal's query (and, if a
// relative guardrail is configured, the same query rendered for the
// baseline track), then checks it against the signal's threshold/
// guardrail.
func (e *Evaluator) evaluateSignal(ctx context.Context, sig policy.SignalConfig, track, windowStr string, now time.Time) (SignalResult, error) {
	sr := SignalResult{Name: sig.Name, Optional: sig.Optional}

	q, err := policy.RenderQuery(sig.Query, policy.TemplateContext{Track: track, Window: windowStr})
	if err != nil {
		return sr, fmt.Errorf("render query: %w", err)
	}
	sr.Query = q

	sample, err := e.Querier.InstantQuery(ctx, q, now)
	if errors.Is(err, metrics.ErrNoData) {
		sr.Missing = true
		sr.ReasonCodes = append(sr.ReasonCodes, ReasonMissingTelemetry)
		return sr, nil
	}
	if err != nil {
		return sr, fmt.Errorf("query: %w", err)
	}
	sr.Value = sample.Value
	sr.HasValue = true
	sr.ThresholdPass = evalThreshold(sig.Threshold, sample.Value)
	if !sr.ThresholdPass {
		sr.ReasonCodes = append(sr.ReasonCodes, ReasonThresholdBreached)
	}

	if sig.Relative != nil {
		baseQ, err := policy.RenderQuery(sig.Query, policy.TemplateContext{Track: e.Policy.Evaluation.BaselineTrack, Window: windowStr})
		if err != nil {
			return sr, fmt.Errorf("render baseline query: %w", err)
		}
		sr.BaselineQuery = baseQ

		baseSample, err := e.Querier.InstantQuery(ctx, baseQ, now)
		if errors.Is(err, metrics.ErrNoData) {
			sr.ReasonCodes = append(sr.ReasonCodes, ReasonMissingTelemetry)
			if sr.ThresholdPass {
				// No independently-known failure yet: a relative guardrail
				// with no trustworthy baseline cannot be evaluated --
				// treating it as "pass" would silently ignore the
				// guardrail, and treating it as "fail" would blame the
				// canary for the baseline's own missing data. Neither is
				// honest, so the whole signal is reported missing (rule 1:
				// never silently treat missing data as healthy).
				sr.Missing = true
				sr.HasValue = false
				return sr, nil
			}
			// The canary already independently breaches its absolute
			// threshold (ADR 0007: absolute thresholds first). That
			// verdict stands on its own regardless of whether the
			// baseline is available -- discarding a known breach just
			// because the *relative* guardrail couldn't be checked would
			// turn a rollback-worthy DEGRADED signal into a falsely
			// lenient INCONCLUSIVE one. The missing baseline is still
			// recorded in ReasonCodes above so the audit trail shows the
			// relative guardrail was never evaluated this window.
			return sr, nil
		}
		if err != nil {
			return sr, fmt.Errorf("baseline query: %w", err)
		}
		sr.BaselineValue = baseSample.Value
		sr.HasBaseline = true

		pass := evalRelative(*sig.Relative, sample.Value, baseSample.Value)
		sr.RelativePass = &pass
		if !pass {
			sr.ReasonCodes = append(sr.ReasonCodes, ReasonRelativeBreached)
		}
	}

	return sr, nil
}

func evalThreshold(t policy.Threshold, v float64) bool {
	if t.Max != nil && v > *t.Max {
		return false
	}
	if t.Min != nil && v < *t.Min {
		return false
	}
	return true
}

// evalRelative checks canary against baseline per the configured ratio
// bounds. baseline <= 0 is handled explicitly to avoid a divide-by-zero
// producing a meaningless ratio: a zero (or negative, which should not
// occur for these signals but is defended against anyway) baseline with a
// nonzero canary is an undefined-but-clearly-large ratio, so it fails a
// MaxRatio bound and passes a MinRatio bound; a zero baseline with a zero
// canary is the degenerate 0/0 case, treated as within bounds for both.
func evalRelative(r policy.RelativeGuardrail, canary, baseline float64) bool {
	if r.MaxRatio != nil {
		if baseline <= 0 {
			if canary > 0 {
				return false
			}
		} else if canary/baseline > *r.MaxRatio {
			return false
		}
	}
	if r.MinRatio != nil {
		// baseline <= 0 always passes a MinRatio bound: a nonzero canary
		// against a zero baseline is an undefined-but-clearly-large ratio
		// (canary/baseline -> +Inf, which is >= any finite MinRatio), and a
		// zero canary against a zero baseline is the degenerate 0/0 case,
		// also treated as within bounds (see doc comment above).
		if baseline > 0 && canary/baseline < *r.MinRatio {
			return false
		}
	}
	return true
}
