package rollout

import "context"

// Action is the effect a rollout decision has on the world outside the
// state machine: actually promoting or rolling back the canary. Phase 5
// (this package) depends only on this interface, never on a concrete
// Kubernetes client -- Phase 6's real adapter (patching stable/canary
// replica counts, per ADR 0003) will implement it, and everything in this
// package is already tested against it today via NoopAction and
// RecordingAction below.
//
// Implementations MUST be idempotent (Claude Code handoff rule 2): the
// state machine's own transition table already guarantees Promote/
// Rollback is called at most once per rollout in the normal case (see
// state.go), but an implementation should not rely on that alone -- a
// Phase 6 reconcile loop recovering from a restart may re-derive and
// re-apply the "same" action, and that must be a safe no-op.
type Action interface {
	Promote(ctx context.Context, rolloutID string) error
	Rollback(ctx context.Context, rolloutID string) error
}

// NoopAction is an Action that does nothing and never fails. It is
// trivially idempotent (doing nothing, twice, is still nothing) and is
// the default Action a Rollout uses when none is supplied (see New) --
// appropriate for Phase 5's simulated rollouts, where the point is
// exercising the state machine itself, not a real deployment side effect.
type NoopAction struct{}

func (NoopAction) Promote(context.Context, string) error  { return nil }
func (NoopAction) Rollback(context.Context, string) error { return nil }

// RecordingAction wraps another Action and counts how many times
// Promote/Rollback were called, for tests asserting idempotency (an
// action must be invoked at most once per rollout even if RecordWindow is
// called again after a terminal state is reached).
type RecordingAction struct {
	Inner         Action
	PromoteCalls  int
	RollbackCalls int
}

func (r *RecordingAction) Promote(ctx context.Context, rolloutID string) error {
	r.PromoteCalls++
	if r.Inner != nil {
		return r.Inner.Promote(ctx, rolloutID)
	}
	return nil
}

func (r *RecordingAction) Rollback(ctx context.Context, rolloutID string) error {
	r.RollbackCalls++
	if r.Inner != nil {
		return r.Inner.Rollback(ctx, rolloutID)
	}
	return nil
}

// FailingAction always returns Err from both methods -- used to test the
// Promoting/RollingBack -> Failed path (an adapter failure, distinct from
// an evaluator/evidence problem).
type FailingAction struct {
	Err error
}

func (f FailingAction) Promote(context.Context, string) error  { return f.Err }
func (f FailingAction) Rollback(context.Context, string) error { return f.Err }
