package dependency

import (
	"errors"
	"sync"
	"testing"
)

// stubSource returns a fixed value from Float64, letting tests place the
// simulated dependency on either side of the errorRate threshold
// deterministically.
type stubSource struct{ value float64 }

func (s stubSource) Float64() float64 { return s.value }

func TestCall_Success(t *testing.T) {
	m := newWithSource(stubSource{value: 0.99})
	if err := m.Call(0.5, false); err != nil {
		t.Fatalf("Call() = %v, want nil (rng above error rate)", err)
	}
}

func TestCall_TransientFailure(t *testing.T) {
	m := newWithSource(stubSource{value: 0.01})
	err := m.Call(0.5, false)
	if !errors.Is(err, ErrTransient) {
		t.Fatalf("Call() = %v, want ErrTransient (rng below error rate)", err)
	}
}

func TestCall_BoundaryIsSuccess(t *testing.T) {
	// rng == errorRate should NOT count as a failure (strict less-than),
	// so errorRate=0 never fails even given a zero draw.
	m := newWithSource(stubSource{value: 0.5})
	if err := m.Call(0.5, false); err != nil {
		t.Fatalf("Call() = %v, want nil at exact boundary", err)
	}
}

func TestCall_ZeroErrorRateNeverFails(t *testing.T) {
	m := newWithSource(stubSource{value: 0.0})
	if err := m.Call(0.0, false); err != nil {
		t.Fatalf("Call() = %v, want nil when errorRate is 0", err)
	}
}

func TestCall_ForcedDownOverridesErrorRate(t *testing.T) {
	// Even with an rng draw that would otherwise succeed, down=true must
	// always fail -- this is the "hard outage" scenario and must never be
	// silently masked by a low error rate.
	m := newWithSource(stubSource{value: 0.99})
	err := m.Call(0.0, true)
	if !errors.Is(err, ErrDown) {
		t.Fatalf("Call() = %v, want ErrDown when down=true", err)
	}
}

// TestCall_ConcurrentUseIsRaceFree exercises a single shared Mock (the
// server package holds exactly one, constructed once in server.New, and
// calls it from every /work request goroutine) from many goroutines at
// once with a nonzero error rate, which is the only code path that
// actually touches the underlying math/rand.Rand and therefore the only
// path that can race. Run with `-race`: before Mock serialized access to
// rng with mu, this reliably tripped the race detector because
// math/rand.Rand (unlike the top-level math/rand functions) is not safe
// for concurrent use.
func TestCall_ConcurrentUseIsRaceFree(t *testing.T) {
	m := New(1)
	const goroutines = 50
	const callsEach = 20

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < callsEach; j++ {
				_ = m.Call(0.5, false)
			}
		}()
	}
	wg.Wait()
}

func TestNew_DeterministicWithSameSeed(t *testing.T) {
	a := New(42)
	b := New(42)
	for i := 0; i < 20; i++ {
		errA := a.Call(0.5, false)
		errB := b.Call(0.5, false)
		if (errA == nil) != (errB == nil) {
			t.Fatalf("iteration %d: same seed produced diverging outcomes: %v vs %v", i, errA, errB)
		}
	}
}
