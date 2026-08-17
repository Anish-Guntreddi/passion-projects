// Package rollout implements ReleaseGuard's release lifecycle state
// machine (Phase 5, FR4): PENDING -> DEPLOYING -> OBSERVING ->
// HEALTHY/DEGRADED/INCONCLUSIVE -> PROMOTING/ROLLING_BACK/PAUSED ->
// COMPLETED/FAILED, driven one window at a time by internal/evaluator's
// WindowResults, with every transition persisted through internal/audit.
//
// See docs/decisions/0010-missing-telemetry-and-pause-semantics.md for the
// two non-obvious design choices this package makes: what OnMissingPause
// actually does across repeated windows, and what "PAUSED -> COMPLETED"
// means in an MVP with no human-in-the-loop resume API yet.
package rollout

// State is one node in the rollout lifecycle state machine. String-valued
// (not iota) so it serializes directly into audit.Event.FromState/ToState
// without a lookup table.
type State string

const (
	StatePending      State = "PENDING"
	StateDeploying    State = "DEPLOYING"
	StateObserving    State = "OBSERVING"
	StateHealthy      State = "HEALTHY"
	StateDegraded     State = "DEGRADED"
	StateInconclusive State = "INCONCLUSIVE"
	StatePromoting    State = "PROMOTING"
	StateRollingBack  State = "ROLLING_BACK"
	StatePaused       State = "PAUSED"
	StateCompleted    State = "COMPLETED"
	StateFailed       State = "FAILED"
)

// Decision is the rollout-level verdict (FR4: "Decision states: PROMOTE,
// PAUSE, ROLLBACK, INCONCLUSIVE"), recorded once a rollout leaves the
// observing loop. Note there is no way to reach StatePromoting without
// Decision being DecisionPromote -- see allowedTransitions and rule 1.
type Decision string

const (
	DecisionPromote      Decision = "PROMOTE"
	DecisionPause        Decision = "PAUSE"
	DecisionRollback     Decision = "ROLLBACK"
	DecisionInconclusive Decision = "INCONCLUSIVE"
)

// observingStates is every state RecordWindow may be called from -- the
// "currently observing" umbrella the spec's chain groups as
// "OBSERVING -> HEALTHY/DEGRADED/INCONCLUSIVE".
var observingStates = map[State]bool{
	StateObserving:    true,
	StateHealthy:      true,
	StateDegraded:     true,
	StateInconclusive: true,
}

// allowedTransitions is the single source of truth for every legal state
// change, directly encoding the spec's chain:
//
//	PENDING -> DEPLOYING -> OBSERVING -> HEALTHY/DEGRADED/INCONCLUSIVE ->
//	PROMOTING/ROLLING_BACK/PAUSED -> COMPLETED/FAILED
//
// It is deliberately exhaustive and restrictive: any transition not
// listed here is rejected by transitionLocked, which is what makes rule 1
// ("never auto-promote with missing evidence") a structural property of
// the machine rather than a runtime check an evaluator bug could bypass --
// StateInconclusive has no path to StatePromoting, full stop, and a test
// (state_test.go) asserts exactly that over every entry in this map.
var allowedTransitions = map[State]map[State]bool{
	StatePending:   {StateDeploying: true, StateFailed: true},
	StateDeploying: {StateObserving: true, StateFailed: true},
	// StateObserving -> StatePaused is the max-rollout-duration deadline
	// firing: a guard on the observation loop itself, independent of any
	// single window's classification (the loop always returns to
	// StateObserving between windows -- see RecordWindow -- so this is the
	// state the deadline check is evaluated from). Everything else out of
	// StateObserving is a per-window classification.
	StateObserving:    {StateHealthy: true, StateDegraded: true, StateInconclusive: true, StatePaused: true, StateFailed: true},
	StateHealthy:      {StateObserving: true, StatePromoting: true},
	StateDegraded:     {StateObserving: true, StateRollingBack: true},
	StateInconclusive: {StateObserving: true, StatePaused: true, StateRollingBack: true},
	StatePromoting:    {StateCompleted: true, StateFailed: true},
	StateRollingBack:  {StateCompleted: true, StateFailed: true},
	StatePaused:       {StateCompleted: true},
	StateCompleted:    {},
	StateFailed:       {},
}

// terminal reports whether s has no outgoing transitions.
func terminal(s State) bool {
	return len(allowedTransitions[s]) == 0
}
