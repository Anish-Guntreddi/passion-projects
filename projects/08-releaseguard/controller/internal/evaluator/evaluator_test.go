package evaluator_test

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"releaseguard/controller/internal/evaluator"
	"releaseguard/controller/internal/metrics"
	"releaseguard/controller/internal/policy"
)

var now = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func f64(v float64) *float64 { return &v }

// testPolicy builds a small, realistic 2-signal policy (error_rate,
// latency_p95) with a sample-count gate and a relative guardrail on
// error_rate, mirroring controller/policy/examples/demo-service-default.yaml
// closely enough to exercise every evaluator code path without pulling in
// the full default policy's signal count.
func testPolicy(t *testing.T) *policy.Policy {
	t.Helper()
	p := &policy.Policy{
		Name:    "test-policy",
		Version: 1,
		Evaluation: policy.EvaluationConfig{
			Interval:                    policy.MustParseDuration("15s"),
			Window:                      policy.MustParseDuration("30s"),
			MinSampleCount:              20,
			SampleCountQuery:            `sum(increase(http_requests_total{track="{{.Track}}"}[{{.Window}}]))`,
			ConsecutiveHealthyWindows:   3,
			ConsecutiveUnhealthyWindows: 2,
			MaxRolloutDuration:          policy.MustParseDuration("30m"),
			OnMissingTelemetry:          policy.OnMissingPause,
			BaselineTrack:               "stable",
		},
		Signals: []policy.SignalConfig{
			{
				Name:      "error_rate",
				Query:     `sum(rate(http_requests_total{status=~"5..",track="{{.Track}}"}[{{.Window}}])) / sum(rate(http_requests_total{track="{{.Track}}"}[{{.Window}}]))`,
				Threshold: policy.Threshold{Max: f64(0.02)},
				Relative:  &policy.RelativeGuardrail{MaxRatio: f64(3.0)},
			},
			{
				Name:      "latency_p95",
				Query:     `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{track="{{.Track}}"}[{{.Window}}])) by (le))`,
				Threshold: policy.Threshold{Max: f64(0.3)},
			},
		},
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("test policy invalid: %v", err)
	}
	return p
}

func render(t *testing.T, tmpl, track, window string) string {
	t.Helper()
	q, err := policy.RenderQuery(tmpl, policy.TemplateContext{Track: track, Window: window})
	if err != nil {
		t.Fatalf("render query: %v", err)
	}
	return q
}

func TestEvaluate_Healthy(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.005, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.004, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Healthy {
		t.Fatalf("Classification = %s, want HEALTHY (reasons=%v, signals=%+v)", wr.Classification, wr.ReasonCodes, wr.SignalResults)
	}
	if wr.SampleCount == nil || *wr.SampleCount != 50 {
		t.Errorf("SampleCount = %v, want 50", wr.SampleCount)
	}
}

func TestEvaluate_DegradedByAbsoluteThreshold(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.001, now)
	// latency well over the 0.3 max threshold -> DEGRADED.
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.6, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Degraded {
		t.Fatalf("Classification = %s, want DEGRADED", wr.Classification)
	}
	var latencyResult *evaluator.SignalResult
	for i := range wr.SignalResults {
		if wr.SignalResults[i].Name == "latency_p95" {
			latencyResult = &wr.SignalResults[i]
		}
	}
	if latencyResult == nil || latencyResult.ThresholdPass {
		t.Fatalf("latency_p95 threshold pass = %+v, want a threshold failure", latencyResult)
	}
}

func TestEvaluate_DegradedByRelativeGuardrail(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	// Both within the 0.02 absolute max, but canary is 5x stable -- over
	// the 3.0 max_ratio relative guardrail (ADR 0007: additive check).
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.010, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.002, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Degraded {
		t.Fatalf("Classification = %s, want DEGRADED (relative guardrail breach)", wr.Classification)
	}
}

func TestEvaluate_InconclusiveOnInsufficientSamples(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 5, now) // below MinSampleCount=20
	// Signals deliberately not populated -- the evaluator must stop at the
	// sample-count gate and never even query them.
	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Inconclusive {
		t.Fatalf("Classification = %s, want INCONCLUSIVE", wr.Classification)
	}
	if len(wr.ReasonCodes) != 1 || wr.ReasonCodes[0] != evaluator.ReasonInsufficientSamples {
		t.Errorf("ReasonCodes = %v, want [%s]", wr.ReasonCodes, evaluator.ReasonInsufficientSamples)
	}
	if len(wr.SignalResults) != 0 {
		t.Errorf("SignalResults = %+v, want none queried after failing the sample-count gate", wr.SignalResults)
	}
}

func TestEvaluate_InconclusiveOnMissingSampleCountTelemetry(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture() // sample_count_query left entirely unanswered
	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Inconclusive {
		t.Fatalf("Classification = %s, want INCONCLUSIVE", wr.Classification)
	}
	if wr.ReasonCodes[0] != evaluator.ReasonMissingTelemetry {
		t.Errorf("ReasonCodes = %v, want starting with %s", wr.ReasonCodes, evaluator.ReasonMissingTelemetry)
	}
}

func TestEvaluate_InconclusiveOnMissingSignal_NeverHealthy(t *testing.T) {
	// This is the direct test of Claude Code handoff rule 1: a missing
	// required signal must never be classified HEALTHY, regardless of how
	// healthy every other signal looks.
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.001, now)
	// latency_p95 query intentionally NOT set -> ErrNoData -> missing.

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Inconclusive {
		t.Fatalf("Classification = %s, want INCONCLUSIVE when a required signal is missing", wr.Classification)
	}
	if wr.Classification == evaluator.Healthy {
		t.Fatal("a missing required signal was classified HEALTHY -- rule 1 violation")
	}
}

func TestEvaluate_MissingBaselineMakesRelativeSignalMissing(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.001, now)
	// stable's error_rate NOT set -> baseline missing.
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Inconclusive {
		t.Fatalf("Classification = %s, want INCONCLUSIVE (missing baseline must not silently pass the relative guardrail)", wr.Classification)
	}
}

func TestEvaluate_OptionalSignalMissingDoesNotBlockHealthy(t *testing.T) {
	p := testPolicy(t)
	p.Signals = append(p.Signals, policy.SignalConfig{
		Name:      "business_success_rate",
		Query:     `sum(rate(business_success_total{track="{{.Track}}"}[{{.Window}}]))`,
		Threshold: policy.Threshold{Min: f64(0.9)},
		Optional:  true,
	})
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)
	// business_success_rate intentionally left unset (missing) -- optional.

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Healthy {
		t.Fatalf("Classification = %s, want HEALTHY (a missing *optional* signal must not force INCONCLUSIVE)", wr.Classification)
	}
	found := false
	for _, sr := range wr.SignalResults {
		if sr.Name == "business_success_rate" {
			found = true
			if !sr.Missing {
				t.Error("business_success_rate.Missing = false, want true")
			}
		}
	}
	if !found {
		t.Error("business_success_rate not present in SignalResults")
	}
}

func TestEvaluate_PresentButFailingOptionalSignalDegradesWindow(t *testing.T) {
	// policy.SignalConfig.Optional's doc comment: "A present-but-failing
	// optional signal still degrades the window like any other; only
	// *missing* data is treated leniently, and only for this signal
	// specifically." Only the missing case had a test before; this is the
	// present-but-failing case.
	p := testPolicy(t)
	p.Signals = append(p.Signals, policy.SignalConfig{
		Name:      "business_success_rate",
		Query:     `sum(rate(business_success_total{track="{{.Track}}"}[{{.Window}}]))`,
		Threshold: policy.Threshold{Min: f64(0.9)},
		Optional:  true,
	})
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.001, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)
	// business_success_rate present but far below its 0.9 min -- a hard
	// failure, not missing data.
	fx.Set(render(t, p.Signals[2].Query, "canary", "30s"), 0.1, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Degraded {
		t.Fatalf("Classification = %s, want DEGRADED (present-but-failing optional signal must still degrade the window)", wr.Classification)
	}
}

func TestEvaluate_MissingBaselineDoesNotMaskAbsoluteBreach(t *testing.T) {
	// ADR 0007: absolute thresholds are checked first and stand on their
	// own. A signal that independently breaches its absolute threshold
	// must classify DEGRADED even if its relative guardrail's baseline
	// happens to be unavailable this window -- a missing baseline must
	// never downgrade a known breach to a falsely lenient INCONCLUSIVE.
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	// error_rate clearly breaches its 0.02 absolute max...
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.09, now)
	// ...and its baseline (stable) is NOT set -> baseline missing.
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Degraded {
		t.Fatalf("Classification = %s, want DEGRADED (absolute breach must stand despite missing relative baseline)", wr.Classification)
	}
	var errRateResult *evaluator.SignalResult
	for i := range wr.SignalResults {
		if wr.SignalResults[i].Name == "error_rate" {
			errRateResult = &wr.SignalResults[i]
		}
	}
	if errRateResult == nil {
		t.Fatal("error_rate not present in SignalResults")
	}
	if errRateResult.Missing {
		t.Error("error_rate.Missing = true, want false (the independent absolute breach must not be discarded as missing)")
	}
	if errRateResult.ThresholdPass {
		t.Error("error_rate.ThresholdPass = true, want false")
	}
}

func TestEvaluate_DegradedReasonCodesReflectActualCause(t *testing.T) {
	// A window degraded purely by a relative-guardrail breach (absolute
	// threshold passing) must not claim ABSOLUTE_THRESHOLD_BREACHED --
	// the audit trail must say what actually happened (FR4).
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	// Both within the 0.02 absolute max, but canary is 5x stable.
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.010, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.002, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if wr.Classification != evaluator.Degraded {
		t.Fatalf("Classification = %s, want DEGRADED", wr.Classification)
	}
	for _, rc := range wr.ReasonCodes {
		if rc == evaluator.ReasonThresholdBreached {
			t.Errorf("ReasonCodes = %v, must not contain %s when only the relative guardrail failed", wr.ReasonCodes, evaluator.ReasonThresholdBreached)
		}
	}
	found := false
	for _, rc := range wr.ReasonCodes {
		if rc == evaluator.ReasonRelativeBreached {
			found = true
		}
	}
	if !found {
		t.Errorf("ReasonCodes = %v, want to contain %s", wr.ReasonCodes, evaluator.ReasonRelativeBreached)
	}
}

func TestEvaluate_QuerierErrorPropagates(t *testing.T) {
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.SetError(render(t, p.Signals[0].Query, "canary", "30s"), errors.New("prometheus: connection refused"))

	_, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
	if err == nil {
		t.Fatal("Evaluate returned nil error for a genuine querier failure, want a propagated error")
	}
}

func TestEvalRelative_ZeroBaselineEdgeCases(t *testing.T) {
	p := testPolicy(t)

	newFixtureWith := func(canaryErr, stableErr float64) *metrics.Fixture {
		fx := metrics.NewFixture()
		fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
		fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)
		fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), canaryErr, now)
		fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), stableErr, now)
		return fx
	}

	t.Run("zero baseline zero canary passes", func(t *testing.T) {
		fx := newFixtureWith(0, 0)
		wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if wr.Classification != evaluator.Healthy {
			t.Fatalf("Classification = %s, want HEALTHY (0/0 baseline case)", wr.Classification)
		}
	})

	t.Run("zero baseline nonzero canary fails max_ratio", func(t *testing.T) {
		fx := newFixtureWith(0.001, 0)
		wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if wr.Classification != evaluator.Degraded {
			t.Fatalf("Classification = %s, want DEGRADED (nonzero canary vs zero baseline breaches max_ratio)", wr.Classification)
		}
	})
}

func TestEvalRelative_MinRatioZeroBaselineEdgeCases(t *testing.T) {
	// Mirrors TestEvalRelative_ZeroBaselineEdgeCases but for a MinRatio
	// guardrail specifically: the doc comment on evalRelative says a zero
	// baseline with a zero canary is "the degenerate 0/0 case, treated as
	// within bounds for both" (both MaxRatio and MinRatio) -- this must
	// hold for MinRatio too, not just MaxRatio.
	p := testPolicy(t)
	p.Signals[0].Relative = &policy.RelativeGuardrail{MinRatio: f64(0.5)}
	if err := p.Validate(); err != nil {
		t.Fatalf("test policy invalid: %v", err)
	}

	newFixtureWith := func(canaryErr, stableErr float64) *metrics.Fixture {
		fx := metrics.NewFixture()
		fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
		fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)
		fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), canaryErr, now)
		fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), stableErr, now)
		return fx
	}

	t.Run("zero baseline zero canary passes min_ratio", func(t *testing.T) {
		fx := newFixtureWith(0, 0)
		wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if wr.Classification != evaluator.Healthy {
			t.Fatalf("Classification = %s, want HEALTHY (0/0 baseline case must pass min_ratio too)", wr.Classification)
		}
	})

	t.Run("zero baseline nonzero canary passes min_ratio", func(t *testing.T) {
		fx := newFixtureWith(0.001, 0)
		wr, err := evaluator.New(p, fx).Evaluate(context.Background(), "canary", now)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if wr.Classification != evaluator.Healthy {
			t.Fatalf("Classification = %s, want HEALTHY (nonzero canary vs zero baseline is an undefined-but-large ratio, which passes min_ratio)", wr.Classification)
		}
	})
}

func TestEvaluate_DeterministicAcrossRepeatedRuns(t *testing.T) {
	// Same fixture, same instant -- run twice, results must be identical.
	// This is the "recorded telemetry fixtures produce deterministic
	// decisions" exit criterion, checked directly: no hidden randomness,
	// no dependence on wall-clock time.Now().
	p := testPolicy(t)
	fx := metrics.NewFixture()
	fx.Set(render(t, p.Evaluation.SampleCountQuery, "canary", "30s"), 50, now)
	fx.Set(render(t, p.Signals[0].Query, "canary", "30s"), 0.005, now)
	fx.Set(render(t, p.Signals[0].Query, "stable", "30s"), 0.004, now)
	fx.Set(render(t, p.Signals[1].Query, "canary", "30s"), 0.05, now)

	e := evaluator.New(p, fx)
	first, err := e.Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate (1st): %v", err)
	}
	second, err := e.Evaluate(context.Background(), "canary", now)
	if err != nil {
		t.Fatalf("Evaluate (2nd): %v", err)
	}
	if first.Classification != second.Classification {
		t.Fatalf("non-deterministic classification: %s vs %s", first.Classification, second.Classification)
	}
	if len(first.SignalResults) != len(second.SignalResults) {
		t.Fatalf("non-deterministic signal count: %d vs %d", len(first.SignalResults), len(second.SignalResults))
	}
	for i := range first.SignalResults {
		if !reflect.DeepEqual(first.SignalResults[i], second.SignalResults[i]) {
			t.Errorf("non-deterministic signal[%d]: %+v vs %+v", i, first.SignalResults[i], second.SignalResults[i])
		}
	}
}
