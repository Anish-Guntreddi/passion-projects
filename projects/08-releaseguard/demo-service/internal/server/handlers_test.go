package server

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"releaseguard/demo-service/internal/config"
)

// newTestServer builds a Server with sane, fast-for-testing defaults, then
// applies overrides. Latency knobs default to zero so unit tests don't pay
// real wall-clock sleep time. opts is forwarded to New unchanged -- most
// tests pass none (getting the default no-op tracer), but tests asserting
// on span content (tracing_test.go) pass WithTracerProvider with an
// in-memory recorder.
func newTestServer(t *testing.T, overrides func(*config.Config), opts ...Option) *Server {
	t.Helper()
	cfg := config.Config{
		Port:             8080,
		ReleaseTrack:     config.TrackStable,
		ReleaseVersion:   "v-test",
		BaseLatencyMS:    0,
		LatencyJitterMS:  0,
		ExtraLatencyMS:   0,
		ErrorRate:        0.0,
		DependencyDown:   false,
		MemoryPressureMB: 0,
	}
	if overrides != nil {
		overrides(&cfg)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return New(cfg, logger, opts...)
}

func decodeJSON[T any](t *testing.T, body io.Reader) T {
	t.Helper()
	var v T
	if err := json.NewDecoder(body).Decode(&v); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}
	return v
}

func TestHandleHealth_OK(t *testing.T) {
	srv := newTestServer(t, nil)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	got := decodeJSON[healthResponse](t, rec.Body)
	if got.Status != "ok" {
		t.Errorf("Status = %q, want ok", got.Status)
	}
	if got.Track != "stable" {
		t.Errorf("Track = %q, want stable", got.Track)
	}
	if got.Dependency != "ok" {
		t.Errorf("Dependency = %q, want ok", got.Dependency)
	}
}

func TestHandleHealth_ReportsDegradedDependencyButStays200(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) { c.ErrorRate = 0.5 })
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (liveness must not fail on SLO degradation)", rec.Code, http.StatusOK)
	}
	got := decodeJSON[healthResponse](t, rec.Body)
	if got.Dependency != "degraded" {
		t.Errorf("Dependency = %q, want degraded", got.Dependency)
	}
}

func TestHandleHealth_ReportsDownDependencyButStays200(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) { c.DependencyDown = true })
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	got := decodeJSON[healthResponse](t, rec.Body)
	if got.Dependency != "down" {
		t.Errorf("Dependency = %q, want down", got.Dependency)
	}
}

func TestHandleVersion(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) {
		c.ReleaseVersion = "v2.0.0-canary"
		c.ReleaseTrack = config.TrackCanary
	})
	req := httptest.NewRequest(http.MethodGet, "/version", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	got := decodeJSON[versionResponse](t, rec.Body)
	if got.ReleaseVersion != "v2.0.0-canary" {
		t.Errorf("ReleaseVersion = %q, want v2.0.0-canary", got.ReleaseVersion)
	}
	if got.Track != "canary" {
		t.Errorf("Track = %q, want canary", got.Track)
	}
	if got.ImageVersion == "" {
		t.Errorf("ImageVersion should not be empty (falls back to package default)")
	}
}

func TestHandleWork_SucceedsWhenErrorRateZero(t *testing.T) {
	srv := newTestServer(t, nil)
	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	got := decodeJSON[workResponse](t, rec.Body)
	if got.Result != "ok" {
		t.Errorf("Result = %q, want ok", got.Result)
	}
}

func TestHandleWork_FailsWhenErrorRateOne(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) { c.ErrorRate = 1.0 })
	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d, body=%s", rec.Code, http.StatusServiceUnavailable, rec.Body.String())
	}
	got := decodeJSON[workErrorResponse](t, rec.Body)
	if got.Reason != "transient" {
		t.Errorf("Reason = %q, want transient", got.Reason)
	}
}

func TestHandleWork_FailsWhenDependencyDown(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) {
		c.DependencyDown = true
		// Even a zero error rate must not mask a hard outage.
		c.ErrorRate = 0.0
	})
	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}
	got := decodeJSON[workErrorResponse](t, rec.Body)
	if got.Reason != "down" {
		t.Errorf("Reason = %q, want down", got.Reason)
	}
}

func TestHandleWork_RespectsExtraLatency(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) { c.ExtraLatencyMS = 30 })
	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	got := decodeJSON[workResponse](t, rec.Body)
	if got.LatencyMS < 30 {
		t.Errorf("LatencyMS = %d, want >= 30 (ExtraLatencyMS not honored)", got.LatencyMS)
	}
}

func TestHandleWork_CancelledContextAbortsWithoutPanic(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) { c.BaseLatencyMS = 500 })
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the request is even served
	req := httptest.NewRequest(http.MethodGet, "/work", nil).WithContext(ctx)
	rec := httptest.NewRecorder()

	// Should return promptly without panicking even though the simulated
	// delay (500ms) is much longer than the (already-elapsed) cancellation.
	srv.Handler().ServeHTTP(rec, req)
}

func TestMetricsEndpoint_ExposesKnownMetricNames(t *testing.T) {
	// ErrorRate=1.0 guarantees the dependency-failure counter has at least
	// one observed label combination -- an unobserved CounterVec label set
	// emits nothing in Prometheus exposition format, so a purely healthy
	// request would never populate work_dependency_failures_total.
	srv := newTestServer(t, func(c *config.Config) { c.ErrorRate = 1.0 })

	workReq := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), workReq)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	body := rec.Body.String()
	for _, name := range []string{
		"http_requests_total",
		"http_request_duration_seconds",
		"work_dependency_failures_total",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("/metrics output missing %q", name)
		}
	}
}
