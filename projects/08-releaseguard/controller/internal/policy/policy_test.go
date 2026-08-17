package policy_test

import (
	"strings"
	"testing"

	"releaseguard/controller/internal/policy"
)

// validPolicy returns a minimal, valid policy struct, used as a base that
// individual validation tests mutate one field at a time -- this keeps
// each test case isolated to the single thing it's checking, rather than
// re-deriving a whole policy document per case.
func validPolicy() policy.Policy {
	max := 0.02
	return policy.Policy{
		Name:    "unit-test-policy",
		Version: 1,
		Evaluation: policy.EvaluationConfig{
			Interval:                    policy.MustParseDuration("15s"),
			Window:                      policy.MustParseDuration("30s"),
			ConsecutiveHealthyWindows:   3,
			ConsecutiveUnhealthyWindows: 2,
			MaxRolloutDuration:          policy.MustParseDuration("30m"),
			OnMissingTelemetry:          policy.OnMissingPause,
			BaselineTrack:               "stable",
		},
		Signals: []policy.SignalConfig{
			{
				Name:      "error_rate",
				Query:     `sum(rate(http_requests_total{status=~"5..",track="{{.Track}}"}[{{.Window}}]))`,
				Threshold: policy.Threshold{Max: &max},
			},
		},
	}
}

func TestLoad_YAML(t *testing.T) {
	p, err := policy.Load("testdata/valid.yaml")
	if err != nil {
		t.Fatalf("Load(valid.yaml) returned error: %v", err)
	}
	if p.Name != "testdata-canary" {
		t.Errorf("Name = %q, want %q", p.Name, "testdata-canary")
	}
	if len(p.Signals) != 2 {
		t.Fatalf("len(Signals) = %d, want 2", len(p.Signals))
	}
	if p.Evaluation.Window.Duration.String() != "30s" {
		t.Errorf("Window = %v, want 30s", p.Evaluation.Window.Duration)
	}
	if p.Evaluation.OnMissingTelemetry != policy.OnMissingPause {
		t.Errorf("OnMissingTelemetry = %q, want %q", p.Evaluation.OnMissingTelemetry, policy.OnMissingPause)
	}
}

func TestLoad_JSON(t *testing.T) {
	p, err := policy.Load("testdata/valid.json")
	if err != nil {
		t.Fatalf("Load(valid.json) returned error: %v", err)
	}
	if p.Name != "testdata-canary-json" {
		t.Errorf("Name = %q, want %q", p.Name, "testdata-canary-json")
	}
	if p.Evaluation.OnMissingTelemetry != policy.OnMissingRollback {
		t.Errorf("OnMissingTelemetry = %q, want %q", p.Evaluation.OnMissingTelemetry, policy.OnMissingRollback)
	}
}

func TestLoad_MissingFile(t *testing.T) {
	if _, err := policy.Load("testdata/does-not-exist.yaml"); err == nil {
		t.Fatal("Load(missing file) returned nil error, want an error")
	}
}

func TestParse_ApplyDefaults(t *testing.T) {
	// min_sample_count omitted (defaults to 0, no gate); on_missing_telemetry
	// and baseline_track omitted entirely -- Parse must fill in the safe
	// defaults (pause, stable) rather than rejecting the document.
	doc := `
name: defaults-test
version: 1
evaluation:
  interval: 10s
  window: 20s
  consecutive_healthy_windows: 1
  consecutive_unhealthy_windows: 1
  max_rollout_duration: 5m
signals:
  - name: error_rate
    query: sum(rate(http_requests_total{track="{{.Track}}"}[{{.Window}}]))
    threshold:
      max: 0.1
`
	p, err := policy.Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse returned error: %v", err)
	}
	if p.Evaluation.OnMissingTelemetry != policy.OnMissingPause {
		t.Errorf("default OnMissingTelemetry = %q, want %q (the safe default)", p.Evaluation.OnMissingTelemetry, policy.OnMissingPause)
	}
	if p.Evaluation.BaselineTrack != "stable" {
		t.Errorf("default BaselineTrack = %q, want %q", p.Evaluation.BaselineTrack, "stable")
	}
}

func TestParse_UnknownFieldRejected(t *testing.T) {
	// UnmarshalStrict should reject typos/unknown fields rather than
	// silently ignoring them -- a misspelled "consecutive_helthy_windows"
	// should fail loudly, not fall back to the zero value.
	doc := `
name: strict-test
version: 1
evaluation:
  interval: 10s
  window: 20s
  consecutive_helthy_windows: 3
  consecutive_unhealthy_windows: 2
  max_rollout_duration: 5m
signals:
  - name: error_rate
    query: sum(rate(http_requests_total{track="{{.Track}}"}[{{.Window}}]))
    threshold:
      max: 0.1
`
	if _, err := policy.Parse([]byte(doc)); err == nil {
		t.Fatal("Parse(doc with unknown field) returned nil error, want an error")
	}
}

func TestValidate_TableDriven(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*policy.Policy)
		wantErr string // substring expected in the error message
	}{
		{
			name:    "missing name",
			mutate:  func(p *policy.Policy) { p.Name = "" },
			wantErr: "name is required",
		},
		{
			name:    "wrong version",
			mutate:  func(p *policy.Policy) { p.Version = 2 },
			wantErr: "version must be 1",
		},
		{
			name:    "zero interval",
			mutate:  func(p *policy.Policy) { p.Evaluation.Interval = policy.Duration{} },
			wantErr: "evaluation.interval must be > 0",
		},
		{
			name:    "zero window",
			mutate:  func(p *policy.Policy) { p.Evaluation.Window = policy.Duration{} },
			wantErr: "evaluation.window must be > 0",
		},
		{
			name: "min_sample_count without a query",
			mutate: func(p *policy.Policy) {
				p.Evaluation.MinSampleCount = 10
				p.Evaluation.SampleCountQuery = ""
			},
			wantErr: "sample_count_query is required",
		},
		{
			name:    "consecutive_healthy_windows zero",
			mutate:  func(p *policy.Policy) { p.Evaluation.ConsecutiveHealthyWindows = 0 },
			wantErr: "consecutive_healthy_windows must be >= 1",
		},
		{
			name:    "consecutive_unhealthy_windows negative",
			mutate:  func(p *policy.Policy) { p.Evaluation.ConsecutiveUnhealthyWindows = -1 },
			wantErr: "consecutive_unhealthy_windows must be >= 1",
		},
		{
			name:    "max_rollout_duration shorter than window",
			mutate:  func(p *policy.Policy) { p.Evaluation.MaxRolloutDuration = policy.MustParseDuration("1s") },
			wantErr: "max_rollout_duration must be >= evaluation.window",
		},
		{
			name: "on_missing_telemetry invalid value (no promote escape hatch)",
			mutate: func(p *policy.Policy) {
				p.Evaluation.OnMissingTelemetry = "promote"
			},
			wantErr: `on_missing_telemetry must be "pause" or "rollback"`,
		},
		{
			name:    "on_missing_telemetry empty after zero-value struct",
			mutate:  func(p *policy.Policy) { p.Evaluation.OnMissingTelemetry = "" },
			wantErr: "on_missing_telemetry must be",
		},
		{
			name:    "empty baseline track",
			mutate:  func(p *policy.Policy) { p.Evaluation.BaselineTrack = "  " },
			wantErr: "baseline_track must not be empty",
		},
		{
			name:    "no signals",
			mutate:  func(p *policy.Policy) { p.Signals = nil },
			wantErr: "at least one signal is required",
		},
		{
			name:    "signal missing name",
			mutate:  func(p *policy.Policy) { p.Signals[0].Name = "" },
			wantErr: "signals[0].name is required",
		},
		{
			name: "duplicate signal names",
			mutate: func(p *policy.Policy) {
				p.Signals = append(p.Signals, p.Signals[0])
			},
			wantErr: "is duplicated",
		},
		{
			name:    "signal missing query",
			mutate:  func(p *policy.Policy) { p.Signals[0].Query = "" },
			wantErr: "signals[0]",
		},
		{
			name: "signal query missing {{.Track}}",
			mutate: func(p *policy.Policy) {
				p.Signals[0].Query = `sum(rate(http_requests_total{track="canary"}[{{.Window}}]))`
			},
			wantErr: "must reference {{.Track}}",
		},
		{
			name:    "signal query malformed template",
			mutate:  func(p *policy.Policy) { p.Signals[0].Query = `sum(rate(x{track="{{.Track}"}[{{.Window}}]))` },
			wantErr: "signals[0]",
		},
		{
			name: "signal threshold has neither max nor min",
			mutate: func(p *policy.Policy) {
				p.Signals[0].Threshold = policy.Threshold{}
			},
			wantErr: "threshold must set max and/or min",
		},
		{
			name: "signal threshold min greater than max",
			mutate: func(p *policy.Policy) {
				lo, hi := 0.9, 0.1
				p.Signals[0].Threshold = policy.Threshold{Min: &lo, Max: &hi}
			},
			wantErr: "threshold.min",
		},
		{
			name: "relative guardrail with neither ratio set",
			mutate: func(p *policy.Policy) {
				p.Signals[0].Relative = &policy.RelativeGuardrail{}
			},
			wantErr: "relative must set max_ratio and/or min_ratio",
		},
		{
			name: "relative guardrail negative max_ratio",
			mutate: func(p *policy.Policy) {
				neg := -1.0
				p.Signals[0].Relative = &policy.RelativeGuardrail{MaxRatio: &neg}
			},
			wantErr: "relative.max_ratio must be > 0",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := validPolicy()
			tc.mutate(&p)
			err := p.Validate()
			if err == nil {
				t.Fatalf("Validate() returned nil error, want one containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Validate() error = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestValidate_AggregatesMultipleErrors(t *testing.T) {
	p := validPolicy()
	p.Name = ""
	p.Version = 99
	p.Signals = nil

	err := p.Validate()
	if err == nil {
		t.Fatal("Validate() returned nil error, want one aggregating 3 problems")
	}
	for _, want := range []string{"name is required", "version must be 1", "at least one signal is required"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("aggregated error %q missing expected substring %q", err.Error(), want)
		}
	}
}

func TestValidate_Valid(t *testing.T) {
	p := validPolicy()
	if err := p.Validate(); err != nil {
		t.Fatalf("Validate() on a well-formed policy returned error: %v", err)
	}
}

func TestApplyDefaults_DoesNotOverrideExplicitValues(t *testing.T) {
	p := validPolicy()
	p.Evaluation.OnMissingTelemetry = policy.OnMissingRollback
	p.Evaluation.BaselineTrack = "canary-baseline"
	p.ApplyDefaults()
	if p.Evaluation.OnMissingTelemetry != policy.OnMissingRollback {
		t.Errorf("ApplyDefaults overrode an explicit OnMissingTelemetry: got %q", p.Evaluation.OnMissingTelemetry)
	}
	if p.Evaluation.BaselineTrack != "canary-baseline" {
		t.Errorf("ApplyDefaults overrode an explicit BaselineTrack: got %q", p.Evaluation.BaselineTrack)
	}
}
