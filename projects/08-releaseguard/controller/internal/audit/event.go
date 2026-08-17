// Package audit implements ReleaseGuard's append-only audit trail (ADR
// 0005, D4): every rollout state transition is recorded as one Event,
// answering FR4's required question, "why did ReleaseGuard make this
// decision?" internal/rollout is the only producer of Events; this
// package only knows how to record and read them back, not what a
// rollout or a decision is.
package audit

import "encoding/json"

// Event is one recorded state transition. RolloutID + Timestamp +
// FromState + ToState is the minimum needed to reconstruct a rollout's
// full history in order; ReasonCodes and Evidence are what make that
// history *explain itself* rather than just being a bare state log.
type Event struct {
	RolloutID string       `json:"rollout_id"`
	Timestamp string       `json:"timestamp"` // RFC3339Nano; string (not time.Time) so JSONL round-trips byte-identically
	EventType string       `json:"event_type"`
	FromState string       `json:"from_state"`
	ToState   string       `json:"to_state"`
	Decision  string       `json:"decision,omitempty"`
	Reasons   []string     `json:"reason_codes,omitempty"`
	Evidence  EvidenceJSON `json:"evidence,omitempty"`
}

// EvidenceJSON carries a pre-marshaled evaluator.WindowResult (or any
// other evidence payload) as raw JSON. audit deliberately does not import
// internal/evaluator -- it is the foundational, dependency-free package
// every other package's audit trail flows through, and typing Evidence as
// json.RawMessage keeps it that way while still round-tripping the full
// structured evidence, not just a string summary.
type EvidenceJSON = json.RawMessage

// Logger is the audit sink internal/rollout writes every transition to.
// JSONLLogger (jsonl.go) is the production implementation (ADR 0005);
// InMemoryLogger (memory.go) backs tests.
type Logger interface {
	Record(Event) error
}
