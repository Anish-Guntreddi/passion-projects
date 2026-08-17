package rollout_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"releaseguard/controller/internal/audit"
	"releaseguard/controller/internal/evaluator"
	"releaseguard/controller/internal/policy"
	"releaseguard/controller/internal/rollout"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func testPolicy(consecutiveHealthy, consecutiveUnhealthy int, maxDuration time.Duration, onMissing policy.MissingTelemetryAction) *policy.Policy {
	return &policy.Policy{
		Name:    "rollout-test-policy",
		Version: 1,
		Evaluation: policy.EvaluationConfig{
			Interval:                    policy.MustParseDuration("15s"),
			Window:                      policy.MustParseDuration("30s"),
			ConsecutiveHealthyWindows:   consecutiveHealthy,
			ConsecutiveUnhealthyWindows: consecutiveUnhealthy,
			MaxRolloutDuration:          policy.Duration{Duration: maxDuration},
			OnMissingTelemetry:          onMissing,
			BaselineTrack:               "stable",
		},
	}
}

func window(cls evaluator.Classification, at time.Time) evaluator.WindowResult {
	return evaluator.WindowResult{Timestamp: at, Track: "canary", Classification: cls}
}

func startAndDeploy(t *testing.T, r *rollout.Rollout, now time.Time) {
	t.Helper()
	if err := r.Start(context.Background(), now); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := r.MarkDeployed(context.Background(), now); err != nil {
		t.Fatalf("MarkDeployed: %v", err)
	}
}

// --- Transition-table structural invariants -------------------------------

func TestAllowedTransitions_InconclusiveNeverReachesPromoting(t *testing.T) {
	// The direct structural test of Claude Code handoff rule 1: no
	// sequence of legal transitions can reach StatePromoting from
	// StateInconclusive.
	p := testPolicy(1, 1, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	startAndDeploy(t, r, t0)

	if _, err := r.RecordWindow(context.Background(), window(evaluator.Inconclusive, t0.Add(15*time.Second)), t0.Add(15*time.Second)); err != nil {
		t.Fatalf("RecordWindow: %v", err)
	}
	if r.State() == rollout.StatePromoting {
		t.Fatal("reached StatePromoting from an inconclusive window")
	}
	if r.State() != rollout.StatePaused {
		t.Fatalf("State() = %s, want PAUSED (consecutive_unhealthy_windows=1)", r.State())
	}
	if r.FinalDecision() == rollout.DecisionPromote {
		t.Fatal("FinalDecision() = PROMOTE after inconclusive telemetry -- rule 1 violation")
	}
}

func TestRecordWindow_RejectedBeforeMarkDeployed(t *testing.T) {
	p := testPolicy(1, 1, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	// Start(), but not MarkDeployed(): still DEPLOYING, not an observing state.
	if err := r.Start(context.Background(), t0); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if _, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, t0), t0); err == nil {
		t.Fatal("RecordWindow before MarkDeployed returned nil error")
	}
}

func TestStart_NotCallableTwice(t *testing.T) {
	p := testPolicy(1, 1, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	if err := r.Start(context.Background(), t0); err != nil {
		t.Fatalf("Start (1st): %v", err)
	}
	if err := r.Start(context.Background(), t0); err == nil {
		t.Fatal("Start (2nd) returned nil error, want an error (PENDING -> DEPLOYING is not legal from DEPLOYING)")
	}
}

// --- Scenario A: healthy canary promotes -----------------------------------

func TestRecordWindow_ConsecutiveHealthyPromotes(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	logger := audit.NewInMemoryLogger()
	r := rollout.New("r1", p, action, logger)
	startAndDeploy(t, r, t0)

	now := t0
	for i := 0; i < 2; i++ {
		now = now.Add(15 * time.Second)
		state, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now)
		if err != nil {
			t.Fatalf("RecordWindow[%d]: %v", i, err)
		}
		if state != rollout.StateObserving {
			t.Fatalf("RecordWindow[%d] state = %s, want OBSERVING (streak not yet met)", i, state)
		}
	}
	now = now.Add(15 * time.Second)
	state, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now)
	if err != nil {
		t.Fatalf("RecordWindow[final]: %v", err)
	}
	if state != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", state)
	}
	if r.FinalDecision() != rollout.DecisionPromote {
		t.Fatalf("FinalDecision() = %s, want PROMOTE", r.FinalDecision())
	}
	if action.PromoteCalls != 1 {
		t.Errorf("PromoteCalls = %d, want 1", action.PromoteCalls)
	}
	if action.RollbackCalls != 0 {
		t.Errorf("RollbackCalls = %d, want 0", action.RollbackCalls)
	}

	events := logger.Events()
	if len(events) == 0 {
		t.Fatal("no audit events recorded")
	}
	if events[0].ToState != string(rollout.StateDeploying) {
		t.Errorf("events[0].ToState = %s, want DEPLOYING", events[0].ToState)
	}
	last := events[len(events)-1]
	if last.ToState != string(rollout.StateCompleted) {
		t.Errorf("last event ToState = %s, want COMPLETED", last.ToState)
	}
}

// --- Scenario B/C: degraded canary rolls back ------------------------------

func TestRecordWindow_ConsecutiveDegradedRollsBack(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	now := t0.Add(15 * time.Second)
	if _, err := r.RecordWindow(context.Background(), window(evaluator.Degraded, now), now); err != nil {
		t.Fatalf("RecordWindow[0]: %v", err)
	}
	now = now.Add(15 * time.Second)
	state, err := r.RecordWindow(context.Background(), window(evaluator.Degraded, now), now)
	if err != nil {
		t.Fatalf("RecordWindow[1]: %v", err)
	}
	if state != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED", state)
	}
	if r.FinalDecision() != rollout.DecisionRollback {
		t.Fatalf("FinalDecision() = %s, want ROLLBACK", r.FinalDecision())
	}
	if action.RollbackCalls != 1 {
		t.Errorf("RollbackCalls = %d, want 1", action.RollbackCalls)
	}
	if action.PromoteCalls != 0 {
		t.Errorf("PromoteCalls = %d, want 0", action.PromoteCalls)
	}
}

// --- Scenario E: a transient blip does not overreact -----------------------

func TestRecordWindow_TransientBlipDoesNotRollback(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	now := t0
	seq := []evaluator.Classification{evaluator.Healthy, evaluator.Degraded, evaluator.Healthy, evaluator.Healthy, evaluator.Healthy}
	var last rollout.State
	for i, cls := range seq {
		now = now.Add(15 * time.Second)
		state, err := r.RecordWindow(context.Background(), window(cls, now), now)
		if err != nil {
			t.Fatalf("RecordWindow[%d]: %v", i, err)
		}
		last = state
	}
	if last != rollout.StateCompleted {
		t.Fatalf("final state = %s, want COMPLETED (single blip must not have triggered rollback)", last)
	}
	if r.FinalDecision() != rollout.DecisionPromote {
		t.Fatalf("FinalDecision() = %s, want PROMOTE (3 consecutive healthy windows followed the blip)", r.FinalDecision())
	}
	if action.RollbackCalls != 0 {
		t.Errorf("RollbackCalls = %d, want 0 (the blip alone must not have crossed consecutive_unhealthy_windows=2)", action.RollbackCalls)
	}
}

// --- Scenario D: missing telemetry never promotes ---------------------------

func TestRecordWindow_PersistentMissingTelemetryPauses(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	now := t0
	for i := 0; i < 2; i++ {
		now = now.Add(15 * time.Second)
		if _, err := r.RecordWindow(context.Background(), window(evaluator.Inconclusive, now), now); err != nil {
			t.Fatalf("RecordWindow[%d]: %v", i, err)
		}
	}
	if r.State() != rollout.StatePaused {
		t.Fatalf("State() = %s, want PAUSED", r.State())
	}
	if r.FinalDecision() != rollout.DecisionPause {
		t.Fatalf("FinalDecision() = %s, want PAUSE", r.FinalDecision())
	}
	if action.PromoteCalls != 0 || action.RollbackCalls != 0 {
		t.Errorf("action calls = promote:%d rollback:%d, want 0/0 (pausing takes no rollout action)", action.PromoteCalls, action.RollbackCalls)
	}

	if err := r.Finalize(context.Background(), now.Add(time.Second)); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	if r.State() != rollout.StateCompleted {
		t.Fatalf("State() after Finalize = %s, want COMPLETED", r.State())
	}
	if r.FinalDecision() != rollout.DecisionPause {
		t.Fatalf("FinalDecision() after Finalize = %s, want still PAUSE", r.FinalDecision())
	}
}

func TestRecordWindow_PersistentMissingTelemetryRollsBackWhenConfigured(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingRollback)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	now := t0
	for i := 0; i < 2; i++ {
		now = now.Add(15 * time.Second)
		if _, err := r.RecordWindow(context.Background(), window(evaluator.Inconclusive, now), now); err != nil {
			t.Fatalf("RecordWindow[%d]: %v", i, err)
		}
	}
	if r.State() != rollout.StateCompleted {
		t.Fatalf("State() = %s, want COMPLETED (rollback applied and completed)", r.State())
	}
	if r.FinalDecision() != rollout.DecisionRollback {
		t.Fatalf("FinalDecision() = %s, want ROLLBACK", r.FinalDecision())
	}
	if action.RollbackCalls != 1 {
		t.Errorf("RollbackCalls = %d, want 1", action.RollbackCalls)
	}
}

// --- Deadline: max rollout duration never resolves to promote --------------

func TestRecordWindow_MaxRolloutDurationExceededPauses(t *testing.T) {
	p := testPolicy(3, 2, 20*time.Second, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	// A single HEALTHY window, but delivered after the 20s deadline has
	// already elapsed -- must not promote just because the evidence looked
	// good; the deadline firing means "we ran out of time," not "it's safe."
	now := t0.Add(30 * time.Second)
	state, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now)
	if err != nil {
		t.Fatalf("RecordWindow: %v", err)
	}
	if state != rollout.StatePaused {
		t.Fatalf("state = %s, want PAUSED", state)
	}
	if r.FinalDecision() != rollout.DecisionInconclusive {
		t.Fatalf("FinalDecision() = %s, want INCONCLUSIVE", r.FinalDecision())
	}
	if action.PromoteCalls != 0 {
		t.Fatalf("PromoteCalls = %d, want 0 -- a HEALTHY window after the deadline must never promote", action.PromoteCalls)
	}
}

// --- Idempotency: no duplicate actions after a terminal state --------------

func TestRecordWindow_NoDuplicateActionAfterTerminal(t *testing.T) {
	p := testPolicy(1, 2, time.Hour, policy.OnMissingPause)
	action := &rollout.RecordingAction{}
	r := rollout.New("r1", p, action, nil)
	startAndDeploy(t, r, t0)

	now := t0.Add(15 * time.Second)
	if _, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now); err != nil {
		t.Fatalf("RecordWindow: %v", err)
	}
	if r.State() != rollout.StateCompleted {
		t.Fatalf("State() = %s, want COMPLETED", r.State())
	}
	if action.PromoteCalls != 1 {
		t.Fatalf("PromoteCalls = %d, want 1", action.PromoteCalls)
	}

	// Simulate a duplicate/late reconcile call after the rollout is done.
	for i := 0; i < 3; i++ {
		now = now.Add(15 * time.Second)
		if _, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now); err == nil {
			t.Fatalf("RecordWindow[%d] after COMPLETED returned nil error, want a rejection", i)
		}
	}
	if action.PromoteCalls != 1 {
		t.Errorf("PromoteCalls after repeated post-terminal RecordWindow calls = %d, want still 1", action.PromoteCalls)
	}
}

// --- Action failure -> Failed ------------------------------------------------

func TestRecordWindow_ActionFailureTransitionsToFailed(t *testing.T) {
	p := testPolicy(1, 2, time.Hour, policy.OnMissingPause)
	wantErr := errors.New("kubernetes: connection refused")
	r := rollout.New("r1", p, rollout.FailingAction{Err: wantErr}, nil)
	startAndDeploy(t, r, t0)

	now := t0.Add(15 * time.Second)
	state, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now)
	if err == nil {
		t.Fatal("RecordWindow with a failing action returned nil error")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("error = %v, want wrapping %v", err, wantErr)
	}
	if state != rollout.StateFailed {
		t.Fatalf("state = %s, want FAILED", state)
	}
	if r.Err() == nil {
		t.Error("Err() = nil, want the action's error recorded")
	}
}

// --- Finalize semantics -----------------------------------------------------

func TestFinalize_IdempotentOnAlreadyTerminal(t *testing.T) {
	p := testPolicy(1, 2, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	startAndDeploy(t, r, t0)
	now := t0.Add(15 * time.Second)
	if _, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now); err != nil {
		t.Fatalf("RecordWindow: %v", err)
	}
	if r.State() != rollout.StateCompleted {
		t.Fatalf("State() = %s, want COMPLETED", r.State())
	}
	if err := r.Finalize(context.Background(), now); err != nil {
		t.Fatalf("Finalize on an already-COMPLETED rollout returned an error: %v", err)
	}
}

func TestFinalize_RejectedFromActiveState(t *testing.T) {
	p := testPolicy(3, 2, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	startAndDeploy(t, r, t0)
	if err := r.Finalize(context.Background(), t0); err == nil {
		t.Fatal("Finalize from OBSERVING returned nil error, want a rejection")
	}
}

// --- Audit trail sanity ------------------------------------------------------

func TestRecordWindow_AuditTrailHasEvidence(t *testing.T) {
	p := testPolicy(1, 2, time.Hour, policy.OnMissingPause)
	logger := audit.NewInMemoryLogger()
	r := rollout.New("r1", p, nil, logger)
	startAndDeploy(t, r, t0)

	now := t0.Add(15 * time.Second)
	wr := evaluator.WindowResult{
		Timestamp:      now,
		Track:          "canary",
		Classification: evaluator.Healthy,
		ReasonCodes:    []string{evaluator.ReasonAllSignalsHealthy},
	}
	if _, err := r.RecordWindow(context.Background(), wr, now); err != nil {
		t.Fatalf("RecordWindow: %v", err)
	}

	events := logger.Events()
	var found bool
	for _, e := range events {
		if e.ToState == string(rollout.StateHealthy) {
			found = true
			if len(e.Evidence) == 0 {
				t.Error("HEALTHY transition event has no evidence attached")
			}
			if len(e.Reasons) == 0 {
				t.Error("HEALTHY transition event has no reason codes attached")
			}
			if e.RolloutID != "r1" {
				t.Errorf("event.RolloutID = %q, want r1", e.RolloutID)
			}
		}
	}
	if !found {
		t.Fatal("no HEALTHY transition event recorded")
	}
}

func TestRollout_DefaultsToNoopActionAndInMemoryLogger(t *testing.T) {
	p := testPolicy(1, 1, time.Hour, policy.OnMissingPause)
	r := rollout.New("r1", p, nil, nil)
	startAndDeploy(t, r, t0)
	now := t0.Add(15 * time.Second)
	state, err := r.RecordWindow(context.Background(), window(evaluator.Healthy, now), now)
	if err != nil {
		t.Fatalf("RecordWindow with nil action/logger: %v", err)
	}
	if state != rollout.StateCompleted {
		t.Fatalf("state = %s, want COMPLETED", state)
	}
}
