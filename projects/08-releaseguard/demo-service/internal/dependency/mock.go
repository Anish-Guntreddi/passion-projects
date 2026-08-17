// Package dependency simulates the "small data dependency" the demo
// service's /work handler calls out to. It exists purely so failure
// scenarios (elevated error rate, hard outage) can be reproduced
// deterministically without standing up a real downstream system.
package dependency

import (
	"errors"
	"math/rand"
	"sync"
)

// ErrDown is returned when the dependency has been forced offline
// (config.DependencyDown), independent of the configured error rate.
var ErrDown = errors.New("dependency: forced outage")

// ErrTransient is returned when a simulated call fails according to the
// configured error rate, representing an ordinary transient downstream
// failure (timeout, 5xx, connection reset, ...).
var ErrTransient = errors.New("dependency: simulated transient failure")

// randSource is the minimal surface Mock needs from a random number
// generator. Abstracting it lets tests inject a deterministic stub instead
// of relying on math/rand's global state or a real seed.
type randSource interface {
	Float64() float64
}

// Mock stands in for a downstream data dependency (e.g. a database or
// internal API) that the demo service calls during /work. A single Mock is
// shared by every request the HTTP server handles (net/http runs each
// request on its own goroutine), and math/rand.Rand created via
// rand.New(rand.NewSource(...)) is NOT safe for concurrent use -- only the
// top-level math/rand package functions (backed by a lockable global
// source) are. Call therefore serializes access to rng itself with mu
// rather than relying on callers to coordinate; that keeps Mock safe to
// share across goroutines regardless of how many handlers end up calling
// it, which is the simplest invariant to reason about and verify with
// `go test -race`.
type Mock struct {
	mu  sync.Mutex
	rng randSource
}

// New returns a Mock seeded deterministically from seed. The same seed
// always produces the same sequence of simulated outcomes, which is what
// lets unit tests and demo scripts reproduce a specific failure pattern.
func New(seed int64) *Mock {
	return &Mock{rng: rand.New(rand.NewSource(seed))}
}

// newWithSource is used by tests to inject a stub random source.
func newWithSource(r randSource) *Mock {
	return &Mock{rng: r}
}

// Call simulates one invocation of the downstream dependency.
//
//   - If down is true, Call always returns ErrDown, regardless of errorRate.
//     This models a hard dependency outage (e.g. saturation, crash-looping
//     dependency) as distinct from an elevated-but-nonzero error rate.
//   - Otherwise, Call fails with ErrTransient with probability errorRate
//     (errorRate must already be validated to lie in [0,1] by the caller;
//     config.Load enforces this).
//   - Otherwise Call succeeds (returns nil).
func (m *Mock) Call(errorRate float64, down bool) error {
	if down {
		return ErrDown
	}
	if errorRate > 0 {
		m.mu.Lock()
		draw := m.rng.Float64()
		m.mu.Unlock()
		if draw < errorRate {
			return ErrTransient
		}
	}
	return nil
}
