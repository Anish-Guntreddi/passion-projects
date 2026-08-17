package rollout

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"releaseguard/controller/internal/audit"
	"releaseguard/controller/internal/evaluator"
	"releaseguard/controller/internal/policy"
)

// Rollout is one canary rollout's state machine instance. It is
// deliberately clock-free: every method that can cause a transition takes
// an explicit `now time.Time` rather than calling time.Now() internally,
// so a rollout's entire lifecycle can be driven deterministically and
// instantly in a test (or a CLI simulation, internal/scenario) without
// real sleeps, and so the exact same code path drives both a fast-forward
// simulation and a real-time Phase 6 reconcile loop.
type Rollout struct {
	ID     string
	Policy *policy.Policy
	Action Action
	Audit  audit.Logger

	mu                 sync.Mutex
	state              State
	healthyStreak      int
	degradedStreak     int
	inconclusiveStreak int
	observingSince     time.Time
	finalDecision      Decision
	windowCount        int
	lastErr            error
}

// New builds a Rollout in StatePending. action defaults to NoopAction and
// logger to a fresh InMemoryLogger if nil, so a caller only interested in
// the state machine's logic (not a real Kubernetes effect or a durable
// audit trail) can pass nil, nil.
func New(id string, p *policy.Policy, action Action, logger audit.Logger) *Rollout {
	if action == nil {
		action = NoopAction{}
	}
	if logger == nil {
		logger = audit.NewInMemoryLogger()
	}
	return &Rollout{
		ID:     id,
		Policy: p,
		Action: action,
		Audit:  logger,
		state:  StatePending,
	}
}

// State returns the rollout's current state.
func (r *Rollout) State() State {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.state
}

// FinalDecision returns the decision recorded when the rollout left the
// observing loop (zero value "" if it hasn't yet).
func (r *Rollout) FinalDecision() Decision {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.finalDecision
}

// Err returns the error that caused a transition to StateFailed, if any.
func (r *Rollout) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lastErr
}

// WindowCount returns how many windows RecordWindow has processed.
func (r *Rollout) WindowCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.windowCount
}

// Start transitions PENDING -> DEPLOYING. Valid only once, from PENDING;
// calling it again (e.g. a duplicate reconcile trigger) returns an error
// rather than re-recording a duplicate "started" event -- part of rule 2
// (idempotent rollout actions): Start itself has no external side effect
// yet, but the guard here is the same discipline every other transition
// in this machine follows.
func (r *Rollout) Start(ctx context.Context, now time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.transitionLocked(ctx, StateDeploying, nil, []string{"ROLLOUT_STARTED"}, nil, now)
}

// MarkDeployed transitions DEPLOYING -> OBSERVING and starts the
// max-rollout-duration clock. In Phase 5 this is called directly by the
// simulation driver (internal/scenario) once the (simulated) canary
// deployment is up; Phase 6's real adapter will call it once the
// Kubernetes deployment/rollout status actually reports ready.
func (r *Rollout) MarkDeployed(ctx context.Context, now time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.transitionLocked(ctx, StateObserving, nil, []string{"CANARY_DEPLOYED"}, nil, now); err != nil {
		return err
	}
	r.observingSince = now
	return nil
}

// RecordWindow feeds one evaluator.WindowResult into the state machine.
// Valid only while the rollout is in an observing state (StateObserving,
// or the per-window classification states themselves, which always
// return to StateObserving between windows -- see isObservingLocked).
// Calling it after the rollout has left the observing loop (Promoting,
// RollingBack, Paused, Completed, Failed) is an error and has no effect --
// this is what keeps a duplicate/late reconciliation from ever invoking
// Action.Promote or Action.Rollback a second time (rule 2).
func (r *Rollout) RecordWindow(ctx context.Context, wr evaluator.WindowResult, now time.Time) (State, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !observingStates[r.state] {
		return r.state, fmt.Errorf("rollout %s: RecordWindow called in state %s, which is not an observing state", r.ID, r.state)
	}

	r.windowCount++
	evidence := marshalEvidence(wr)

	if elapsed := now.Sub(r.observingSince); elapsed >= r.Policy.Evaluation.MaxRolloutDuration.Duration {
		// A deadline is an absence of a conclusive decision, not evidence of
		// health -- it can never resolve to PROMOTE (rule 1). Pausing (not
		// failing) is the correct terminal action: nothing crashed, the
		// controller just ran out of time to reach confidence.
		decision := DecisionInconclusive
		return r.transitionToStateLocked(ctx, StatePaused, &decision,
			[]string{"MAX_ROLLOUT_DURATION_EXCEEDED"}, evidence, now)
	}

	switch wr.Classification {
	case evaluator.Healthy:
		r.healthyStreak++
		r.degradedStreak = 0
		r.inconclusiveStreak = 0
		if err := r.transitionLocked(ctx, StateHealthy, nil, wr.ReasonCodes, evidence, now); err != nil {
			return r.state, err
		}
		if r.healthyStreak >= r.Policy.Evaluation.ConsecutiveHealthyWindows {
			decision := DecisionPromote
			if err := r.transitionLocked(ctx, StatePromoting, &decision, []string{"CONSECUTIVE_HEALTHY_WINDOWS_MET"}, evidence, now); err != nil {
				return r.state, err
			}
			return r.applyActionLocked(ctx, r.Action.Promote, now)
		}
		return r.backToObservingLocked(ctx, "AWAITING_MORE_HEALTHY_WINDOWS", evidence, now)

	case evaluator.Degraded:
		r.degradedStreak++
		r.healthyStreak = 0
		r.inconclusiveStreak = 0
		if err := r.transitionLocked(ctx, StateDegraded, nil, wr.ReasonCodes, evidence, now); err != nil {
			return r.state, err
		}
		if r.degradedStreak >= r.Policy.Evaluation.ConsecutiveUnhealthyWindows {
			decision := DecisionRollback
			if err := r.transitionLocked(ctx, StateRollingBack, &decision, []string{"CONSECUTIVE_DEGRADED_WINDOWS_MET"}, evidence, now); err != nil {
				return r.state, err
			}
			return r.applyActionLocked(ctx, r.Action.Rollback, now)
		}
		return r.backToObservingLocked(ctx, "AWAITING_MORE_DEGRADED_WINDOWS", evidence, now)

	case evaluator.Inconclusive:
		r.inconclusiveStreak++
		r.healthyStreak = 0
		r.degradedStreak = 0
		if err := r.transitionLocked(ctx, StateInconclusive, nil, wr.ReasonCodes, evidence, now); err != nil {
			return r.state, err
		}
		if r.inconclusiveStreak >= r.Policy.Evaluation.ConsecutiveUnhealthyWindows {
			return r.resolveMissingTelemetryLocked(ctx, evidence, now)
		}
		return r.backToObservingLocked(ctx, "AWAITING_MORE_WINDOWS", evidence, now)

	default:
		return r.state, fmt.Errorf("rollout %s: unknown classification %q", r.ID, wr.Classification)
	}
}

// resolveMissingTelemetryLocked applies the policy's configured
// on_missing_telemetry action once telemetry has stayed inconclusive for
// ConsecutiveUnhealthyWindows windows in a row. Never promotes -- the
// MissingTelemetryAction enum (policy package) structurally has no
// promote-equivalent value, so this switch cannot reach StatePromoting no
// matter what a policy file says.
func (r *Rollout) resolveMissingTelemetryLocked(ctx context.Context, evidence audit.EvidenceJSON, now time.Time) (State, error) {
	switch r.Policy.Evaluation.OnMissingTelemetry {
	case policy.OnMissingRollback:
		decision := DecisionRollback
		if err := r.transitionLocked(ctx, StateRollingBack, &decision,
			[]string{"PERSISTENT_INCONCLUSIVE_TELEMETRY", "ON_MISSING_TELEMETRY_ROLLBACK"}, evidence, now); err != nil {
			return r.state, err
		}
		return r.applyActionLocked(ctx, r.Action.Rollback, now)
	default: // policy.OnMissingPause, and any unrecognized value defaults safely to pause
		decision := DecisionPause
		return r.transitionToStateLocked(ctx, StatePaused, &decision,
			[]string{"PERSISTENT_INCONCLUSIVE_TELEMETRY", "ON_MISSING_TELEMETRY_PAUSE"}, evidence, now)
	}
}

// backToObservingLocked records the classification -> OBSERVING transition
// that happens whenever a window's streak hasn't yet crossed its
// configured threshold -- the rollout keeps waiting for more evidence.
func (r *Rollout) backToObservingLocked(ctx context.Context, reason string, evidence audit.EvidenceJSON, now time.Time) (State, error) {
	if err := r.transitionLocked(ctx, StateObserving, nil, []string{reason}, evidence, now); err != nil {
		return r.state, err
	}
	return r.state, nil
}

// applyActionLocked invokes action (Promote or Rollback) and records the
// resulting COMPLETED/FAILED transition. Called with the machine already
// in StatePromoting or StateRollingBack, both of which the transition
// table only allows moving on from once -- so even if a caller somehow
// invoked this twice (it can't, through the public API: RecordWindow
// rejects any call once the state has left the observing states), the
// second transitionLocked call would itself fail closed rather than
// re-running the action.
func (r *Rollout) applyActionLocked(ctx context.Context, action func(context.Context, string) error, now time.Time) (State, error) {
	if err := action(ctx, r.ID); err != nil {
		r.lastErr = err
		_ = r.transitionLocked(ctx, StateFailed, nil, []string{"ACTION_FAILED", err.Error()}, nil, now)
		return r.state, fmt.Errorf("rollout %s: action failed: %w", r.ID, err)
	}
	if err := r.transitionLocked(ctx, StateCompleted, nil, []string{"ACTION_APPLIED"}, nil, now); err != nil {
		return r.state, err
	}
	return r.state, nil
}

// Finalize resolves a StatePaused rollout to StateCompleted. Idempotent:
// calling it when already COMPLETED or FAILED is a harmless no-op (returns
// nil, records nothing further); calling it from any other, still-active
// state is an error, since a rollout that is still deploying/observing/
// promoting/rolling back has not reached a stable point to finalize from.
//
// This exists because PAUSED has no automatic path onward in this MVP --
// there is no human-in-the-loop "resume" API yet (see docs/decisions/
// 0010), so the driver (a CLI simulation today, Phase 6's reconcile loop
// later) explicitly finalizes a paused rollout once it has recorded the
// pause, making PAUSED -> COMPLETED an observable, deliberate step in the
// audit trail rather than an implicit one.
func (r *Rollout) Finalize(ctx context.Context, now time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == StateCompleted || r.state == StateFailed {
		return nil
	}
	if r.state != StatePaused {
		return fmt.Errorf("rollout %s: cannot Finalize from state %s (only PAUSED or an already-terminal state)", r.ID, r.state)
	}
	return r.transitionLocked(ctx, StateCompleted, nil, []string{"PAUSED_ROLLOUT_FINALIZED"}, nil, now)
}

// transitionToStateLocked is transitionLocked plus updating finalDecision,
// used by the two paths (deadline exceeded, persistent missing telemetry)
// that go straight to StatePaused with a decision attached.
func (r *Rollout) transitionToStateLocked(ctx context.Context, to State, decision *Decision, reasons []string, evidence audit.EvidenceJSON, now time.Time) (State, error) {
	if err := r.transitionLocked(ctx, to, decision, reasons, evidence, now); err != nil {
		return r.state, err
	}
	return r.state, nil
}

// transitionLocked is the one place r.state actually changes. It enforces
// allowedTransitions (an invalid transition is rejected and r.state is
// left unchanged), then records an audit.Event for the change.
//
// If Audit.Record itself fails (e.g. a full disk), the in-memory state has
// already moved to `to` -- the decision was made correctly and is not
// rolled back on an audit-write failure, but the error is still returned
// so the caller (and its caller, up to whatever's driving the rollout)
// learns the audit trail is not durable right now and can alert/retry.
// Silently swallowing an audit-write failure would defeat FR4's
// auditability requirement more thoroughly than surfacing it noisily.
func (r *Rollout) transitionLocked(ctx context.Context, to State, decision *Decision, reasons []string, evidence audit.EvidenceJSON, now time.Time) error {
	from := r.state
	if !allowedTransitions[from][to] {
		return fmt.Errorf("rollout %s: illegal transition %s -> %s", r.ID, from, to)
	}
	r.state = to
	if decision != nil {
		r.finalDecision = *decision
	}

	event := audit.Event{
		RolloutID: r.ID,
		Timestamp: now.UTC().Format(time.RFC3339Nano),
		EventType: "state_transition",
		FromState: string(from),
		ToState:   string(to),
		Reasons:   reasons,
		Evidence:  evidence,
	}
	if decision != nil {
		event.Decision = string(*decision)
	}

	_ = ctx // reserved for a future context-aware Logger (e.g. tracing); unused today
	if err := r.Audit.Record(event); err != nil {
		return fmt.Errorf("rollout %s: audit record %s -> %s: %w", r.ID, from, to, err)
	}
	return nil
}

// marshalEvidence best-effort marshals a WindowResult for the audit
// event. A marshal failure here (which would require a non-JSON-safe
// value from the evaluator, not expected in practice) degrades to no
// evidence rather than losing the transition record entirely.
func marshalEvidence(wr evaluator.WindowResult) audit.EvidenceJSON {
	b, err := json.Marshal(wr)
	if err != nil {
		return nil
	}
	return b
}
