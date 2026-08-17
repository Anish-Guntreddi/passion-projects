package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"releaseguard/demo-service/internal/config"
)

func TestInstrument_RecordsRequestsTotal(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) {
		c.ReleaseVersion = "v1.0.0"
		c.ReleaseTrack = config.TrackCanary
	})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	got := testutil.ToFloat64(srv.metrics.RequestsTotal.WithLabelValues(
		"/health", http.MethodGet, "200", "v1.0.0", "canary",
	))
	if got != 2 {
		t.Fatalf("http_requests_total{route=/health,...} = %v, want 2", got)
	}
}

func TestInstrument_RecordsErrorStatusLabel(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) {
		c.ErrorRate = 1.0
		c.ReleaseVersion = "v1.0.0"
	})

	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	got := testutil.ToFloat64(srv.metrics.RequestsTotal.WithLabelValues(
		"/work", http.MethodGet, "503", "v1.0.0", "stable",
	))
	if got != 1 {
		t.Fatalf("http_requests_total{route=/work,status=503,...} = %v, want 1", got)
	}
}

func TestInstrument_RecordsDependencyFailureReason(t *testing.T) {
	srv := newTestServer(t, func(c *config.Config) {
		c.DependencyDown = true
		c.ReleaseVersion = "v1.0.0"
	})

	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	got := testutil.ToFloat64(srv.metrics.DependencyFailuresTotal.WithLabelValues(
		"down", "v1.0.0", "stable",
	))
	if got != 1 {
		t.Fatalf("work_dependency_failures_total{reason=down,...} = %v, want 1", got)
	}
}

func TestInstrument_RequestDurationObserved(t *testing.T) {
	srv := newTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	count := testutil.CollectAndCount(srv.metrics.RequestDuration)
	if count == 0 {
		t.Fatalf("expected http_request_duration_seconds to have at least one observed series, got 0")
	}
}
