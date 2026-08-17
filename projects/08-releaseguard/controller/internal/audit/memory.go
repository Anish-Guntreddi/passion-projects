package audit

import "sync"

// InMemoryLogger is a Logger backed by an in-process slice. Used by tests
// (internal/rollout's state-machine tests, internal/scenario's simulation
// tests) that want to assert on the exact recorded event sequence without
// a filesystem dependency, and as the default Logger when none is
// supplied (see rollout.New).
type InMemoryLogger struct {
	mu     sync.Mutex
	events []Event
}

// NewInMemoryLogger returns an empty InMemoryLogger.
func NewInMemoryLogger() *InMemoryLogger {
	return &InMemoryLogger{}
}

func (l *InMemoryLogger) Record(e Event) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, e)
	return nil
}

// Events returns a copy of every event recorded so far, in recording
// order. A copy, not the internal slice, so callers can't mutate the
// logger's state through the returned slice.
func (l *InMemoryLogger) Events() []Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]Event, len(l.events))
	copy(out, l.events)
	return out
}
