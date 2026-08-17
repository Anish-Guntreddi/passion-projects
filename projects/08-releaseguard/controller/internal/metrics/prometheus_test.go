package metrics_test

// Integration-style test for the Prometheus adapter (spec §1.7: "Prometheus
// query adapter"). It runs against an httptest.Server that speaks the real
// Prometheus HTTP API response shape, rather than a live Prometheus
// binary -- enough to exercise the actual HTTP client, URL construction and
// JSON-decoding path end to end without a cluster dependency. Co-located
// with the package under test (controller/internal/metrics) rather than
// under tests/integration/: Go's internal-package visibility rule is keyed
// off import-path prefix, and tests/integration/ lives outside the
// releaseguard/controller module's import path, so it could never import
// releaseguard/controller/internal/metrics regardless of directory
// nesting on disk -- the same reason unit tests are co-located per
// tests/README.md.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"releaseguard/controller/internal/metrics"
)

// promResponse builds a Prometheus HTTP API JSON response body for the
// given resultType/result payload.
func promResponse(t *testing.T, resultType string, result any) []byte {
	t.Helper()
	body := map[string]any{
		"status": "success",
		"data": map[string]any{
			"resultType": resultType,
			"result":     result,
		},
	}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal fake prometheus response: %v", err)
	}
	return b
}

// newServer starts a fake Prometheus API server that returns responses[query]
// verbatim as the response body for both /api/v1/query and
// /api/v1/query_range, keyed on the "query" form value. A query not present
// in responses gets a 200 with an empty vector/matrix result.
func newServer(t *testing.T, responses map[string][]byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	handle := func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("parse form: %v", err)
		}
		q := r.Form.Get("query")
		w.Header().Set("Content-Type", "application/json")
		if body, ok := responses[q]; ok {
			w.Write(body)
			return
		}
		if r.URL.Path == "/api/v1/query_range" {
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
			return
		}
		w.Write([]byte(`{"status":"success","data":{"resultType":"vector","result":[]}}`))
	}
	mux.HandleFunc("/api/v1/query", handle)
	mux.HandleFunc("/api/v1/query_range", handle)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestPrometheus_InstantQuery_Vector(t *testing.T) {
	const q = `sum(rate(http_requests_total{track="canary"}[30s]))`
	ts := time.Now().Truncate(time.Second)
	body := promResponse(t, "vector", []map[string]any{
		{
			"metric": map[string]string{},
			"value":  []any{float64(ts.Unix()), "0.015"},
		},
	})
	srv := newServer(t, map[string][]byte{q: body})

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	sample, err := p.InstantQuery(context.Background(), q, ts)
	if err != nil {
		t.Fatalf("InstantQuery: %v", err)
	}
	if sample.Value != 0.015 {
		t.Errorf("sample.Value = %v, want 0.015", sample.Value)
	}
}

func TestPrometheus_InstantQuery_Scalar(t *testing.T) {
	const q = `1 + 1`
	ts := time.Now().Truncate(time.Second)
	body := promResponse(t, "scalar", []any{float64(ts.Unix()), "2"})
	srv := newServer(t, map[string][]byte{q: body})

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	sample, err := p.InstantQuery(context.Background(), q, ts)
	if err != nil {
		t.Fatalf("InstantQuery: %v", err)
	}
	if sample.Value != 2 {
		t.Errorf("sample.Value = %v, want 2", sample.Value)
	}
}

func TestPrometheus_InstantQuery_EmptyVectorIsErrNoData(t *testing.T) {
	const q = `sum(rate(http_requests_total{track="canary"}[30s]))`
	srv := newServer(t, map[string][]byte{}) // unmatched -> empty vector

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	_, err = p.InstantQuery(context.Background(), q, time.Now())
	if !errors.Is(err, metrics.ErrNoData) {
		t.Fatalf("InstantQuery error = %v, want wrapping ErrNoData", err)
	}
}

func TestPrometheus_InstantQuery_NaNIsErrNoData(t *testing.T) {
	const q = `sum(rate(http_requests_total{track="canary",status=~"5.."}[30s])) / sum(rate(http_requests_total{track="canary"}[30s]))`
	ts := time.Now().Truncate(time.Second)
	body := promResponse(t, "vector", []map[string]any{
		{"metric": map[string]string{}, "value": []any{float64(ts.Unix()), "NaN"}},
	})
	srv := newServer(t, map[string][]byte{q: body})

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	_, err = p.InstantQuery(context.Background(), q, ts)
	if !errors.Is(err, metrics.ErrNoData) {
		t.Fatalf("InstantQuery error = %v, want wrapping ErrNoData (NaN is Prometheus's 0/0)", err)
	}
}

func TestPrometheus_InstantQuery_MultipleSeriesIsError(t *testing.T) {
	const q = `rate(http_requests_total{track="canary"}[30s])`
	ts := time.Now().Truncate(time.Second)
	body := promResponse(t, "vector", []map[string]any{
		{"metric": map[string]string{"route": "/work"}, "value": []any{float64(ts.Unix()), "1"}},
		{"metric": map[string]string{"route": "/health"}, "value": []any{float64(ts.Unix()), "2"}},
	})
	srv := newServer(t, map[string][]byte{q: body})

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	if _, err := p.InstantQuery(context.Background(), q, ts); err == nil {
		t.Fatal("InstantQuery with 2 series returned nil error, want an error demanding aggregation")
	}
}

func TestPrometheus_RangeQuery(t *testing.T) {
	const q = `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{track="canary"}[30s])) by (le))`
	end := time.Now().Truncate(time.Second)
	start := end.Add(-1 * time.Minute)
	body := promResponse(t, "matrix", []map[string]any{
		{
			"metric": map[string]string{},
			"values": []any{
				[]any{float64(start.Unix()), "0.05"},
				[]any{float64(end.Unix()), "0.06"},
			},
		},
	})
	srv := newServer(t, map[string][]byte{q: body})

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	series, err := p.RangeQuery(context.Background(), q, metrics.Range{Start: start, End: end, Step: 15 * time.Second})
	if err != nil {
		t.Fatalf("RangeQuery: %v", err)
	}
	if len(series.Points) != 2 {
		t.Fatalf("len(series.Points) = %d, want 2", len(series.Points))
	}
	if series.Points[0].Value != 0.05 || series.Points[1].Value != 0.06 {
		t.Errorf("series.Points = %+v, want [0.05, 0.06]", series.Points)
	}
}

func TestPrometheus_RangeQuery_EmptyIsErrNoData(t *testing.T) {
	const q = `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{track="canary"}[30s])) by (le))`
	end := time.Now().Truncate(time.Second)
	start := end.Add(-1 * time.Minute)
	srv := newServer(t, map[string][]byte{}) // unmatched -> empty vector, not matrix, so decode via query_range default

	p, err := metrics.NewPrometheus(srv.URL, nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	_, err = p.RangeQuery(context.Background(), q, metrics.Range{Start: start, End: end, Step: 15 * time.Second})
	if !errors.Is(err, metrics.ErrNoData) {
		t.Fatalf("RangeQuery error = %v, want wrapping ErrNoData", err)
	}
}

func TestPrometheus_ConnectionFailure(t *testing.T) {
	// Simulates §1.7's "Prometheus unavailable" failure scenario: no
	// server listening at all.
	p, err := metrics.NewPrometheus("http://127.0.0.1:1", nil)
	if err != nil {
		t.Fatalf("NewPrometheus: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := p.InstantQuery(ctx, "up", time.Now()); err == nil {
		t.Fatal("InstantQuery against an unreachable server returned nil error")
	}
}
