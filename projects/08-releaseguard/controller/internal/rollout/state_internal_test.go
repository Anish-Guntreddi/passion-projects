package rollout

import "testing"

// allStates enumerates every declared State, used to exhaustively check
// allowedTransitions below rather than spot-checking a few entries.
var allStates = []State{
	StatePending, StateDeploying, StateObserving, StateHealthy, StateDegraded,
	StateInconclusive, StatePromoting, StateRollingBack, StatePaused,
	StateCompleted, StateFailed,
}

func TestAllowedTransitions_EveryStateHasATableEntry(t *testing.T) {
	for _, s := range allStates {
		if _, ok := allowedTransitions[s]; !ok {
			t.Errorf("state %s has no entry in allowedTransitions", s)
		}
	}
}

func TestAllowedTransitions_TerminalStatesHaveNoOutgoingEdges(t *testing.T) {
	for _, s := range []State{StateCompleted, StateFailed} {
		if !terminal(s) {
			t.Errorf("state %s has outgoing transitions %v, want none (it's a terminal state)", s, allowedTransitions[s])
		}
	}
}

// TestAllowedTransitions_OnlyHealthyLeadsToPromoting is the exhaustive
// structural proof behind Claude Code handoff rule 1: the only edge in the
// whole table that leads to StatePromoting originates at StateHealthy.
//
// Note this deliberately checks *direct* incoming edges, not general graph
// reachability: StateInconclusive can legally reach StatePromoting
// eventually (StateInconclusive -> StateObserving -> ... -> StateHealthy
// (a *later*, actually-healthy window) -> StatePromoting), and that's
// correct behavior -- one inconclusive blip must not permanently block a
// canary that goes on to earn its consecutive healthy streak from fresh
// evidence. What must never happen is promotion *as a direct consequence*
// of inconclusive or degraded evidence, which is exactly "no non-HEALTHY
// state has an edge straight to PROMOTING."
func TestAllowedTransitions_OnlyHealthyLeadsToPromoting(t *testing.T) {
	for from, tos := range allowedTransitions {
		if tos[StatePromoting] && from != StateHealthy {
			t.Errorf("state %s has a direct transition to PROMOTING; only HEALTHY may", from)
		}
	}
}

func TestAllowedTransitions_PendingReachesEveryTerminalOutcome(t *testing.T) {
	// Sanity check that the graph is actually connected the way the spec's
	// chain describes: from PENDING you can eventually reach COMPLETED (via
	// any of promote/rollback/pause) and FAILED (an action failure at any
	// stage).
	seen := map[State]bool{StatePending: true}
	queue := []State{StatePending}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for next := range allowedTransitions[cur] {
			if !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
	for _, want := range allStates {
		if !seen[want] {
			t.Errorf("state %s is not reachable from PENDING", want)
		}
	}
}
