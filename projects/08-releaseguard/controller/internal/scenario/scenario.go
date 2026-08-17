// Package scenario loads and runs a "recorded telemetry" fixture -- a
// scripted sequence of per-window metric values -- through a real
// Evaluator and Rollout, end to end. It is the single piece of code both
// `cmd/controller simulate` (the human-facing local reproducibility
// command) and the Phase 4/5 exit-criterion tests
// (controller/scenarios/*_test.go) run, so there is exactly one
// implementation of "how a scenario file becomes a rollout" to keep
// correct -- not one for the CLI and a subtly different one for tests.
package scenario

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"releaseguard/controller/internal/audit"
	"releaseguard/controller/internal/evaluator"
	"releaseguard/controller/internal/metrics"
	"releaseguard/controller/internal/policy"
	"releaseguard/controller/internal/rollout"
)

// Tick is one simulated observation window's recorded telemetry: the
// value of each signal (by name, matching a policy.SignalConfig.Name) for
// the canary track and, when a relative guardrail needs it, the baseline
// track. A signal name absent from Canary (or from Stable, for a signal
// with a relative guardrail) means that signal's telemetry is missing for
// this window -- the evaluator will see metrics.ErrNoData for it, exactly
// as it would for a real gap in recorded Prometheus data.
type Tick struct {
	Canary map[string]float64 `json:"canary"`
	Stable map[string]float64 `json:"stable,omitempty"`
	// SampleCount is the canary sample-count-gate value for this window.
	// nil means "no data" (missing telemetry at the gate itself), distinct
	// from a present-but-too-low count.
	SampleCount *float64 `json:"sample_count,omitempty"`
}

// Scenario is a named, recorded-telemetry fixture: spec §1.8's five demo
// scenarios (A-E) each get one file under controller/scenarios/.
type Scenario struct {
	Description string `json:"description,omitempty"`
	Ticks       []Tick `json:"ticks"`
}

// Load reads and parses a scenario file.
func Load(path string) (*Scenario, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("scenario: read %s: %w", path, err)
	}
	var s Scenario
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("scenario: parse %s: %w", path, err)
	}
	if len(s.Ticks) == 0 {
		return nil, fmt.Errorf("scenario: %s: at least one tick is required", path)
	}
	return &s, nil
}

// BuildQuerier renders every query a policy's evaluator would issue for
// tick and returns a metrics.Fixture answering exactly those queries with
// the tick's recorded values -- a signal (or the sample-count gate) not
// present in the tick is deliberately left unanswered, so the evaluator
// sees the same metrics.ErrNoData it would against a real gap in recorded
// Prometheus data.
func BuildQuerier(p *policy.Policy, tick Tick) (*metrics.Fixture, error) {
	windowStr := p.Evaluation.Window.Duration.String()
	fx := metrics.NewFixture()

	if p.Evaluation.MinSampleCount > 0 && tick.SampleCount != nil {
		q, err := policy.RenderQuery(p.Evaluation.SampleCountQuery, policy.TemplateContext{Track: "canary", Window: windowStr})
		if err != nil {
			return nil, fmt.Errorf("scenario: render sample_count_query: %w", err)
		}
		fx.Set(q, *tick.SampleCount, time.Time{})
	}

	for _, sig := range p.Signals {
		if v, ok := tick.Canary[sig.Name]; ok {
			q, err := policy.RenderQuery(sig.Query, policy.TemplateContext{Track: "canary", Window: windowStr})
			if err != nil {
				return nil, fmt.Errorf("scenario: render signal %q query: %w", sig.Name, err)
			}
			fx.Set(q, v, time.Time{})
		}
		if sig.Relative != nil {
			if v, ok := tick.Stable[sig.Name]; ok {
				q, err := policy.RenderQuery(sig.Query, policy.TemplateContext{Track: p.Evaluation.BaselineTrack, Window: windowStr})
				if err != nil {
					return nil, fmt.Errorf("scenario: render signal %q baseline query: %w", sig.Name, err)
				}
				fx.Set(q, v, time.Time{})
			}
		}
	}
	return fx, nil
}

// Result is what running a Scenario produced: the finished Rollout plus
// its full recorded audit trail, for a caller (CLI or test) to inspect or
// print without reaching back into the Rollout's internals.
type Result struct {
	Rollout *rollout.Rollout
	Events  []audit.Event
}

// Run drives a full rollout -- Start, MarkDeployed, then one RecordWindow
// per Tick -- against policy p, stopping as soon as the rollout reaches a
// terminal outcome (COMPLETED/FAILED) or PAUSED (which Run finalizes
// immediately, since this simulation has no human operator to leave it
// waiting on -- see rollout.Rollout.Finalize's doc comment). If the
// scenario's ticks run out before a terminal outcome, Run returns the
// rollout in whatever state it stopped at (a scenario file that never
// resolves is itself a meaningful thing to be able to observe/test).
//
// now is the simulated start time; each tick advances the clock by
// p.Evaluation.Interval, matching a live reconcile loop's cadence exactly
// -- the whole point of a clock-free Rollout (see machine.go) is that this
// same call sequence is what Phase 6's real, real-time loop will make.
func Run(ctx context.Context, id string, p *policy.Policy, action rollout.Action, logger audit.Logger, scen *Scenario, start time.Time) (*Result, error) {
	if logger == nil {
		logger = audit.NewInMemoryLogger()
	}
	r := rollout.New(id, p, action, logger)

	if err := r.Start(ctx, start); err != nil {
		return nil, fmt.Errorf("scenario: Start: %w", err)
	}
	if err := r.MarkDeployed(ctx, start); err != nil {
		return nil, fmt.Errorf("scenario: MarkDeployed: %w", err)
	}

	now := start
	for i, tick := range scen.Ticks {
		now = now.Add(p.Evaluation.Interval.Duration)

		fx, err := BuildQuerier(p, tick)
		if err != nil {
			return nil, fmt.Errorf("scenario: tick %d: %w", i, err)
		}
		wr, err := evaluator.New(p, fx).Evaluate(ctx, "canary", now)
		if err != nil {
			return nil, fmt.Errorf("scenario: tick %d: evaluate: %w", i, err)
		}
		state, err := r.RecordWindow(ctx, wr, now)
		if err != nil {
			return nil, fmt.Errorf("scenario: tick %d: RecordWindow: %w", i, err)
		}

		switch state {
		case rollout.StateCompleted, rollout.StateFailed:
			return result(r, logger), nil
		case rollout.StatePaused:
			if err := r.Finalize(ctx, now); err != nil {
				return nil, fmt.Errorf("scenario: tick %d: Finalize: %w", i, err)
			}
			return result(r, logger), nil
		}
	}
	return result(r, logger), nil
}

func result(r *rollout.Rollout, logger audit.Logger) *Result {
	var events []audit.Event
	if mem, ok := logger.(*audit.InMemoryLogger); ok {
		events = mem.Events()
	}
	return &Result{Rollout: r, Events: events}
}
