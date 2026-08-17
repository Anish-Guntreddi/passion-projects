package scenario_test

// These tests run the five spec §1.8 demo scenarios (A-E) through the
// real default policy (controller/policy/examples/demo-service-default.yaml)
// end to end -- policy parsing, evaluator, and the full rollout state
// machine -- against recorded telemetry fixtures. This is the direct proof
// of Phase 4's exit criterion ("recorded telemetry fixtures produce
// deterministic decisions") and Phase 5's ("simulated rollout executes
// full state sequence"), using the actual policy file a human would
// review, not a synthetic test-only policy.

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"releaseguard/controller/internal/policy"
	"releaseguard/controller/internal/rollout"
	"releaseguard/controller/internal/scenario"
)

const defaultPolicyPath = "../../policy/examples/demo-service-default.yaml"

var simStart = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func loadDefaultPolicy(t *testing.T) *policy.Policy {
	t.Helper()
	p, err := policy.Load(defaultPolicyPath)
	if err != nil {
		t.Fatalf("policy.Load(%s): %v", defaultPolicyPath, err)
	}
	return p
}

func runScenario(t *testing.T, path string) *scenario.Result {
	t.Helper()
	p := loadDefaultPolicy(t)
	scen, err := scenario.Load(path)
	if err != nil {
		t.Fatalf("scenario.Load(%s): %v", path, err)
	}
	action := &rollout.RecordingAction{}
	res, err := scenario.Run(context.Background(), "sim-"+t.Name(), p, action, nil, scen, simStart)
	if err != nil {
		t.Fatalf("scenario.Run(%s): %v", path, err)
	}
	return res
}

func TestScenarioA_HealthyPromotion(t *testing.T) {
	res := runScenario(t, "../../scenarios/a-healthy-promotion.json")
	if res.Rollout.State() != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", res.Rollout.State())
	}
	if res.Rollout.FinalDecision() != rollout.DecisionPromote {
		t.Fatalf("FinalDecision() = %s, want PROMOTE", res.Rollout.FinalDecision())
	}
	if res.Rollout.WindowCount() != 3 {
		t.Errorf("WindowCount() = %d, want 3", res.Rollout.WindowCount())
	}
}

func TestScenarioB_LatencyRegressionRollback(t *testing.T) {
	res := runScenario(t, "../../scenarios/b-latency-regression-rollback.json")
	if res.Rollout.State() != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", res.Rollout.State())
	}
	if res.Rollout.FinalDecision() != rollout.DecisionRollback {
		t.Fatalf("FinalDecision() = %s, want ROLLBACK", res.Rollout.FinalDecision())
	}
}

func TestScenarioC_ErrorRegressionRollback(t *testing.T) {
	res := runScenario(t, "../../scenarios/c-error-regression-rollback.json")
	if res.Rollout.State() != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", res.Rollout.State())
	}
	if res.Rollout.FinalDecision() != rollout.DecisionRollback {
		t.Fatalf("FinalDecision() = %s, want ROLLBACK", res.Rollout.FinalDecision())
	}
}

func TestScenarioD_MissingTelemetryPause(t *testing.T) {
	res := runScenario(t, "../../scenarios/d-missing-telemetry-pause.json")
	if res.Rollout.State() != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED (paused, then finalized)", res.Rollout.State())
	}
	if res.Rollout.FinalDecision() != rollout.DecisionPause {
		t.Fatalf("FinalDecision() = %s, want PAUSE -- missing telemetry must never resolve to PROMOTE", res.Rollout.FinalDecision())
	}
	if res.Rollout.FinalDecision() == rollout.DecisionPromote {
		t.Fatal("missing-telemetry scenario resulted in PROMOTE -- rule 1 violation")
	}

	// The audit trail must actually contain a PAUSED transition -- not
	// just a final decision that happens to say PAUSE.
	var sawPaused bool
	for _, e := range res.Events {
		if e.ToState == string(rollout.StatePaused) {
			sawPaused = true
		}
	}
	if !sawPaused {
		t.Error("audit trail has no PAUSED transition")
	}
}

func TestScenarioE_TransientBlipPromotion(t *testing.T) {
	res := runScenario(t, "../../scenarios/e-transient-blip-promotion.json")
	if res.Rollout.State() != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", res.Rollout.State())
	}
	if res.Rollout.FinalDecision() != rollout.DecisionPromote {
		t.Fatalf("FinalDecision() = %s, want PROMOTE (the blip must not have triggered a rollback)", res.Rollout.FinalDecision())
	}
	if res.Rollout.WindowCount() != 5 {
		t.Errorf("WindowCount() = %d, want 5 (all ticks consumed: blip + 3 fresh healthy windows)", res.Rollout.WindowCount())
	}

	var sawDegraded bool
	for _, e := range res.Events {
		if e.ToState == string(rollout.StateDegraded) {
			sawDegraded = true
		}
	}
	if !sawDegraded {
		t.Error("audit trail has no DEGRADED transition -- the blip should still be visible even though it didn't cause a rollback")
	}
}

// TestScenarios_DeterministicAcrossRepeatedRuns runs every scenario twice
// and asserts an identical outcome both times -- the direct check of
// "recorded telemetry fixtures produce deterministic decisions."
func TestScenarios_DeterministicAcrossRepeatedRuns(t *testing.T) {
	paths := []string{
		"../../scenarios/a-healthy-promotion.json",
		"../../scenarios/b-latency-regression-rollback.json",
		"../../scenarios/c-error-regression-rollback.json",
		"../../scenarios/d-missing-telemetry-pause.json",
		"../../scenarios/e-transient-blip-promotion.json",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			p := loadDefaultPolicy(t)
			scen, err := scenario.Load(path)
			if err != nil {
				t.Fatalf("scenario.Load: %v", err)
			}

			run := func() *scenario.Result {
				res, err := scenario.Run(context.Background(), "sim-determinism", p, &rollout.RecordingAction{}, nil, scen, simStart)
				if err != nil {
					t.Fatalf("scenario.Run: %v", err)
				}
				return res
			}

			first := run()
			second := run()

			if first.Rollout.FinalDecision() != second.Rollout.FinalDecision() {
				t.Fatalf("non-deterministic decision: %s vs %s", first.Rollout.FinalDecision(), second.Rollout.FinalDecision())
			}
			if first.Rollout.State() != second.Rollout.State() {
				t.Fatalf("non-deterministic final state: %s vs %s", first.Rollout.State(), second.Rollout.State())
			}
			if len(first.Events) != len(second.Events) {
				t.Fatalf("non-deterministic event count: %d vs %d", len(first.Events), len(second.Events))
			}
			for i := range first.Events {
				// RolloutID differs by construction (not part of "the
				// decision"); compare everything else.
				a, b := first.Events[i], second.Events[i]
				a.RolloutID, b.RolloutID = "", ""
				if !reflect.DeepEqual(a, b) {
					t.Errorf("event[%d] differs between runs:\n  1st: %+v\n  2nd: %+v", i, a, b)
				}
			}
		})
	}
}

func TestLoad_EmptyTicksRejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(path, []byte(`{"ticks": []}`), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := scenario.Load(path); err == nil {
		t.Fatal("Load(scenario with no ticks) returned nil error")
	}
}

func TestLoad_MissingFile(t *testing.T) {
	if _, err := scenario.Load("does-not-exist.json"); err == nil {
		t.Fatal("Load(missing file) returned nil error")
	}
}
