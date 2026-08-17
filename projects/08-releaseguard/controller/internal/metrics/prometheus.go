package metrics

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"time"

	promapi "github.com/prometheus/client_golang/api"
	promv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
)

// Prometheus is the real Querier implementation: it wraps
// github.com/prometheus/client_golang's generated v1 API client against a
// live Prometheus HTTP API (ADR 0004, D3). Everything the evaluator does
// with a Querier is expressed against the interface in types.go, so this
// type is the only place PromQL result-type handling (vector vs scalar,
// NaN detection, multi-series ambiguity) lives.
type Prometheus struct {
	api promv1.API
}

// NewPrometheus builds a Prometheus querier against addr (e.g.
// "http://prometheus.releaseguard.svc:9090"). rt is the http.RoundTripper
// to use; pass nil for http.DefaultTransport (a caller wanting timeouts,
// TLS config, etc. supplies its own).
func NewPrometheus(addr string, rt http.RoundTripper) (*Prometheus, error) {
	client, err := promapi.NewClient(promapi.Config{Address: addr, RoundTripper: rt})
	if err != nil {
		return nil, fmt.Errorf("metrics: build prometheus client: %w", err)
	}
	return &Prometheus{api: promv1.NewAPI(client)}, nil
}

func (p *Prometheus) InstantQuery(ctx context.Context, query string, ts time.Time) (Sample, error) {
	val, _, err := p.api.Query(ctx, query, ts)
	if err != nil {
		return Sample{}, fmt.Errorf("metrics: prometheus instant query %q: %w", query, err)
	}
	return sampleFromValue(query, val)
}

func (p *Prometheus) RangeQuery(ctx context.Context, query string, r Range) (Series, error) {
	val, _, err := p.api.QueryRange(ctx, query, promv1.Range{Start: r.Start, End: r.End, Step: r.Step})
	if err != nil {
		return Series{}, fmt.Errorf("metrics: prometheus range query %q: %w", query, err)
	}
	matrix, ok := val.(model.Matrix)
	if !ok {
		return Series{}, fmt.Errorf("metrics: prometheus range query %q: unexpected result type %T", query, val)
	}
	switch len(matrix) {
	case 0:
		return Series{}, fmt.Errorf("%w: range query %q returned no series", ErrNoData, query)
	case 1:
		points := make([]Sample, 0, len(matrix[0].Values))
		for _, v := range matrix[0].Values {
			if math.IsNaN(float64(v.Value)) {
				continue
			}
			points = append(points, Sample{Value: float64(v.Value), Timestamp: v.Timestamp.Time()})
		}
		return Series{Points: points}, nil
	default:
		// A policy's queries are expected to already aggregate to a single
		// series (sum(...) / histogram_quantile(...) etc., matching the
		// examples in docs/slo-policy.md) -- more than one series back means
		// the query is under-aggregated relative to what the evaluator
		// expects, which is a policy authoring error, not something to
		// silently pick the "first" series from.
		return Series{}, fmt.Errorf("metrics: range query %q returned %d series, want exactly 1 (aggregate the query, e.g. with sum())", query, len(matrix))
	}
}

// sampleFromValue extracts a single scalar Sample from a Prometheus
// instant-query result, which model.Value represents as either a Vector
// (the common case: an aggregating PromQL expression evaluated at one
// instant) or a Scalar (a bare numeric expression). Any other shape
// (Matrix, String) is not a meaningful "current value of a signal" and is
// treated as a query-authoring error.
func sampleFromValue(query string, val model.Value) (Sample, error) {
	switch v := val.(type) {
	case model.Vector:
		switch len(v) {
		case 0:
			return Sample{}, fmt.Errorf("%w: instant query %q returned no series", ErrNoData, query)
		case 1:
			f := float64(v[0].Value)
			if math.IsNaN(f) {
				return Sample{}, fmt.Errorf("%w: instant query %q returned NaN", ErrNoData, query)
			}
			return Sample{Value: f, Timestamp: v[0].Timestamp.Time()}, nil
		default:
			return Sample{}, fmt.Errorf("metrics: instant query %q returned %d series, want exactly 1 (aggregate the query, e.g. with sum())", query, len(v))
		}
	case *model.Scalar:
		f := float64(v.Value)
		if math.IsNaN(f) {
			return Sample{}, fmt.Errorf("%w: instant query %q returned NaN", ErrNoData, query)
		}
		return Sample{Value: f, Timestamp: v.Timestamp.Time()}, nil
	default:
		return Sample{}, fmt.Errorf("metrics: instant query %q: unexpected result type %T", query, val)
	}
}
