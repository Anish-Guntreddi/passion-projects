package metrics_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"releaseguard/controller/internal/metrics"
)

func TestFixture_InstantQuery_KnownValue(t *testing.T) {
	now := time.Now()
	f := metrics.NewFixture().Set("q1", 0.02, now)
	s, err := f.InstantQuery(context.Background(), "q1", now)
	if err != nil {
		t.Fatalf("InstantQuery: %v", err)
	}
	if s.Value != 0.02 {
		t.Errorf("Value = %v, want 0.02", s.Value)
	}
}

func TestFixture_InstantQuery_UnknownIsErrNoData(t *testing.T) {
	f := metrics.NewFixture()
	_, err := f.InstantQuery(context.Background(), "unknown", time.Now())
	if !errors.Is(err, metrics.ErrNoData) {
		t.Fatalf("InstantQuery(unknown) error = %v, want wrapping ErrNoData", err)
	}
}

func TestFixture_InstantQuery_ExplicitError(t *testing.T) {
	wantErr := errors.New("prometheus unavailable")
	f := metrics.NewFixture().SetError("q1", wantErr)
	_, err := f.InstantQuery(context.Background(), "q1", time.Now())
	if !errors.Is(err, wantErr) {
		t.Fatalf("InstantQuery error = %v, want %v", err, wantErr)
	}
}

func TestFixture_RangeQuery(t *testing.T) {
	series := metrics.Series{Points: []metrics.Sample{{Value: 1}, {Value: 2}}}
	f := metrics.NewFixture().SetSeries("q1", series)
	got, err := f.RangeQuery(context.Background(), "q1", metrics.Range{})
	if err != nil {
		t.Fatalf("RangeQuery: %v", err)
	}
	if len(got.Points) != 2 {
		t.Fatalf("len(Points) = %d, want 2", len(got.Points))
	}
}

func TestFixture_RecordsQueriesAsked(t *testing.T) {
	f := metrics.NewFixture().Set("q1", 1, time.Now())
	_, _ = f.InstantQuery(context.Background(), "q1", time.Now())
	_, _ = f.InstantQuery(context.Background(), "q2", time.Now())
	if len(f.Queries) != 2 || f.Queries[0] != "q1" || f.Queries[1] != "q2" {
		t.Errorf("Queries = %v, want [q1 q2]", f.Queries)
	}
}
