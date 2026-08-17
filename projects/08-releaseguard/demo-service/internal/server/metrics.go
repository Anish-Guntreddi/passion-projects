package server

import "github.com/prometheus/client_golang/prometheus"

// Metrics groups every custom Prometheus collector the demo-service
// exposes. Phase 2 (Telemetry) wires a real Prometheus server to scrape
// these; Phase 0/1 only need the instrumentation points to exist so that
// later phases have "version" and "track" as first-class label values
// (per FR3: latency/errors/version must be queryable per stable/canary
// identity) instead of being retrofitted.
type Metrics struct {
	// RequestsTotal counts every HTTP request the server has handled,
	// labeled by route, method, response status, release version and
	// track (stable/canary).
	RequestsTotal *prometheus.CounterVec

	// RequestDuration observes HTTP request handling latency in seconds,
	// labeled by route, method, release version and track. This is what a
	// canary evaluator would query for P95/P99 comparisons.
	RequestDuration *prometheus.HistogramVec

	// DependencyFailuresTotal counts simulated downstream dependency call
	// failures, labeled by failure reason ("transient" or "down"),
	// release version and track.
	DependencyFailuresTotal *prometheus.CounterVec
}

// NewMetrics constructs and registers the demo-service's custom
// collectors against reg. It panics on duplicate registration (via
// MustRegister), which is the correct failure mode: a metrics-naming
// collision is a programming error that should fail loudly at startup,
// not be swallowed.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total HTTP requests processed, labeled by route, method, status, version and track.",
		}, []string{"route", "method", "status", "version", "track"}),

		RequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds, labeled by route, method, version and track.",
			Buckets: prometheus.DefBuckets,
		}, []string{"route", "method", "version", "track"}),

		DependencyFailuresTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "work_dependency_failures_total",
			Help: "Total simulated dependency call failures, labeled by reason, version and track.",
		}, []string{"reason", "version", "track"}),
	}

	reg.MustRegister(m.RequestsTotal, m.RequestDuration, m.DependencyFailuresTotal)
	return m
}
