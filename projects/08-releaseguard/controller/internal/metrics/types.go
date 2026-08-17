// Package metrics defines ReleaseGuard's metrics-query abstraction (ADR
// 0004, D3): the evaluator depends only on the Querier interface, never on
// a concrete Prometheus client, which is what makes Phase 4's evaluator
// unit-testable against fixture data with zero network or cluster
// dependency (Fixture, in fixture.go) while still having a real
// implementation that talks to a live Prometheus server (Prometheus, in
// prometheus.go, wired up starting Phase 2/6).
package metrics

import (
	"context"
	"errors"
	"time"
)

// ErrNoData is returned by a Querier when a query executed successfully
// but produced no usable value -- no series matched, or the result was
// NaN (Prometheus's own representation of e.g. 0/0 from a rate-over-empty
// division). The evaluator (internal/evaluator) treats ErrNoData as
// missing telemetry, never as a value to compare against a threshold --
// see Claude Code handoff rule 1 ("never auto-promote with missing
// evidence").
var ErrNoData = errors.New("metrics: query returned no data")

// Sample is a single scalar metric value at a point in time.
type Sample struct {
	Value     float64
	Timestamp time.Time
}

// Range describes a range-query window: [Start, End] stepped by Step.
type Range struct {
	Start time.Time
	End   time.Time
	Step  time.Duration
}

// Series is a time-ordered sequence of samples, returned by a range query.
// It backs audit evidence (a graphable trace of "why", not just a
// verdict) rather than the pass/fail decision itself, which is driven by
// instant queries.
type Series struct {
	Points []Sample
}

// Querier is the abstraction the evaluator depends on for both "what is
// the value of this signal right now" (InstantQuery) and "show me this
// signal over the observation window" (RangeQuery) -- ADR 0004's two query
// shapes behind one interface.
type Querier interface {
	// InstantQuery evaluates query at instant ts and returns the single
	// resulting scalar value. Returns ErrNoData (wrapped) if the query
	// produced no series or a NaN value.
	InstantQuery(ctx context.Context, query string, ts time.Time) (Sample, error)

	// RangeQuery evaluates query over r and returns its time series.
	// Returns ErrNoData (wrapped) if the query produced no series.
	RangeQuery(ctx context.Context, query string, r Range) (Series, error)
}
