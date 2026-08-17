package metrics

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Fixture is a Querier backed by canned, recorded data rather than a live
// Prometheus server -- this is what makes the evaluator's tests exercise
// "recorded telemetry fixtures produce deterministic decisions" (Phase 4
// exit criterion) without a network dependency. Queries not present in
// Values/Errors resolve to ErrNoData by default: an evaluator test must
// name every query it expects to be answered, so a template-rendering bug
// (e.g. a query the evaluator didn't mean to send) shows up as "missing
// telemetry" in the test output rather than silently passing.
type Fixture struct {
	mu      sync.Mutex
	Values  map[string]Sample
	Errors  map[string]error
	Series  map[string]Series
	Queries []string // every query string InstantQuery/RangeQuery was asked, in order
}

// NewFixture returns an empty Fixture ready to have Set/SetError/SetSeries
// called on it.
func NewFixture() *Fixture {
	return &Fixture{
		Values: make(map[string]Sample),
		Errors: make(map[string]error),
		Series: make(map[string]Series),
	}
}

// Set records the instant-query value to return for query.
func (f *Fixture) Set(query string, value float64, ts time.Time) *Fixture {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Values[query] = Sample{Value: value, Timestamp: ts}
	return f
}

// SetError records the error InstantQuery/RangeQuery should return for
// query (e.g. a simulated Prometheus-unavailable condition, distinct from
// "no data" -- see §1.7's failure-testing requirement).
func (f *Fixture) SetError(query string, err error) *Fixture {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Errors[query] = err
	return f
}

// SetSeries records the range-query series to return for query.
func (f *Fixture) SetSeries(query string, s Series) *Fixture {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Series[query] = s
	return f
}

func (f *Fixture) InstantQuery(_ context.Context, query string, _ time.Time) (Sample, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Queries = append(f.Queries, query)
	if err, ok := f.Errors[query]; ok {
		return Sample{}, err
	}
	if s, ok := f.Values[query]; ok {
		return s, nil
	}
	return Sample{}, fmt.Errorf("%w: query %q not present in fixture", ErrNoData, query)
}

func (f *Fixture) RangeQuery(_ context.Context, query string, _ Range) (Series, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Queries = append(f.Queries, query)
	if err, ok := f.Errors[query]; ok {
		return Series{}, err
	}
	if s, ok := f.Series[query]; ok {
		return s, nil
	}
	return Series{}, fmt.Errorf("%w: query %q not present in fixture", ErrNoData, query)
}
