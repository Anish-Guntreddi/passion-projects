package policy

import (
	"errors"
	"fmt"
	"strings"
)

// Validate checks every field FR4 requires a typed policy to carry, and
// returns a single error aggregating every problem found (not just the
// first) so a human fixing a policy file sees the whole list in one pass.
// Call ApplyDefaults first (Parse/Load already do).
func (p *Policy) Validate() error {
	var errs []string
	add := func(format string, args ...any) {
		errs = append(errs, fmt.Sprintf(format, args...))
	}

	if strings.TrimSpace(p.Name) == "" {
		add("name is required")
	}
	if p.Version != 1 {
		add("version must be 1, got %d", p.Version)
	}

	p.validateEvaluation(add)
	p.validateSignals(add)

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func (p *Policy) validateEvaluation(add func(string, ...any)) {
	e := p.Evaluation

	if e.Interval.Duration <= 0 {
		add("evaluation.interval must be > 0")
	}
	if e.Window.Duration <= 0 {
		add("evaluation.window must be > 0")
	}
	if e.MinSampleCount < 0 {
		add("evaluation.min_sample_count must be >= 0")
	}
	if e.MinSampleCount > 0 {
		if strings.TrimSpace(e.SampleCountQuery) == "" {
			add("evaluation.sample_count_query is required when min_sample_count > 0")
		} else {
			validateQueryTemplate(add, "evaluation.sample_count_query", e.SampleCountQuery, true)
		}
	}
	if e.ConsecutiveHealthyWindows < 1 {
		add("evaluation.consecutive_healthy_windows must be >= 1")
	}
	if e.ConsecutiveUnhealthyWindows < 1 {
		add("evaluation.consecutive_unhealthy_windows must be >= 1")
	}
	if e.MaxRolloutDuration.Duration <= 0 {
		add("evaluation.max_rollout_duration must be > 0")
	} else if e.Window.Duration > 0 && e.MaxRolloutDuration.Duration < e.Window.Duration {
		add("evaluation.max_rollout_duration must be >= evaluation.window")
	}

	switch e.OnMissingTelemetry {
	case OnMissingPause, OnMissingRollback:
	default:
		// Deliberately does not list a "promote" option in the message --
		// there isn't one. See MissingTelemetryAction's doc comment.
		add("evaluation.on_missing_telemetry must be %q or %q, got %q",
			OnMissingPause, OnMissingRollback, e.OnMissingTelemetry)
	}

	if strings.TrimSpace(e.BaselineTrack) == "" {
		add("evaluation.baseline_track must not be empty")
	}
}

func (p *Policy) validateSignals(add func(string, ...any)) {
	if len(p.Signals) == 0 {
		add("at least one signal is required")
		return
	}

	seen := make(map[string]bool, len(p.Signals))
	for i, s := range p.Signals {
		prefix := fmt.Sprintf("signals[%d]", i)
		name := strings.TrimSpace(s.Name)
		if name == "" {
			add("%s.name is required", prefix)
		} else {
			prefix = fmt.Sprintf("signals[%d] (%s)", i, name)
			if seen[name] {
				add("%s.name %q is duplicated", prefix, name)
			}
			seen[name] = true
		}

		if strings.TrimSpace(s.Query) == "" {
			add("%s.query is required", prefix)
		} else {
			validateQueryTemplate(add, prefix+".query", s.Query, true)
		}

		if s.Threshold.Max == nil && s.Threshold.Min == nil {
			add("%s.threshold must set max and/or min", prefix)
		}
		if s.Threshold.Max != nil && s.Threshold.Min != nil && *s.Threshold.Min > *s.Threshold.Max {
			add("%s.threshold.min (%v) must be <= threshold.max (%v)", prefix, *s.Threshold.Min, *s.Threshold.Max)
		}

		if s.Relative != nil {
			if s.Relative.MaxRatio == nil && s.Relative.MinRatio == nil {
				add("%s.relative must set max_ratio and/or min_ratio", prefix)
			}
			if s.Relative.MaxRatio != nil && *s.Relative.MaxRatio <= 0 {
				add("%s.relative.max_ratio must be > 0", prefix)
			}
			if s.Relative.MinRatio != nil && *s.Relative.MinRatio <= 0 {
				add("%s.relative.min_ratio must be > 0", prefix)
			}
			// A relative guardrail compares canary against baseline: if the
			// query text is track-invariant it would render identically for
			// both, making the comparison trivially 1:1 regardless of real
			// values -- a silent no-op guardrail rather than a config error.
			// Caught here instead of at evaluation time.
			if !strings.Contains(s.Query, "{{.Track}}") {
				add("%s.query must reference {{.Track}} to support a relative guardrail", prefix)
			}
		}
	}
}

// validateQueryTemplate checks that a query template parses and renders
// successfully (catching template syntax errors at policy-load time) and,
// when requireTrack is true, that it actually varies by track.
func validateQueryTemplate(add func(string, ...any), field, tmpl string, requireTrack bool) {
	if requireTrack && !strings.Contains(tmpl, "{{.Track}}") {
		add("%s must reference {{.Track}} so it is scoped to the track being evaluated", field)
	}
	if _, err := RenderQuery(tmpl, TemplateContext{Track: "canary", Window: "30s"}); err != nil {
		add("%s: %s", field, err)
	}
}
