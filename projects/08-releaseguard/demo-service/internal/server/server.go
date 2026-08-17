// Package server implements the demo-service's HTTP surface: /health,
// /version, /work and /metrics. It is deliberately the only application
// under test in Phases 0-1 -- the ReleaseGuard controller itself is built
// starting Phase 4, once there is real stable/canary telemetry to
// evaluate.
package server

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"releaseguard/demo-service/internal/config"
	"releaseguard/demo-service/internal/dependency"
)

// pageSize is used to touch one byte per virtual-memory page of the
// memory-pressure ballast so the OS actually commits the pages (visible in
// RSS / go_memstats) instead of leaving them as an unmapped, zero-cost
// reservation.
const pageSize = 4096

// Server holds the demo-service's dependencies and exposes an
// http.Handler. It is intentionally small: no framework, no global state,
// so every behavior can be traced from an incoming request to the
// config/metric/dependency it touches.
type Server struct {
	cfg       config.Config
	dep       *dependency.Mock
	metrics   *Metrics
	registry  *prometheus.Registry
	startedAt time.Time
	logger    *slog.Logger

	// ballast is memory allocated (and touched) at startup to simulate
	// MEMORY_PRESSURE_MB. It is retained for the lifetime of the Server so
	// the OS cannot reclaim it, and is otherwise unused.
	ballast []byte
}

// New constructs a Server ready to serve traffic. It builds its own
// prometheus.Registry (rather than using the global default registry) so
// that multiple Servers -- one per test, for example -- never collide on
// metric registration.
func New(cfg config.Config, logger *slog.Logger) *Server {
	registry := prometheus.NewRegistry()
	registry.MustRegister(
		prometheus.NewGoCollector(),
		prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}),
	)
	metrics := NewMetrics(registry)

	s := &Server{
		cfg:       cfg,
		dep:       dependency.New(time.Now().UnixNano()),
		metrics:   metrics,
		registry:  registry,
		startedAt: time.Now(),
		logger:    logger,
	}

	if cfg.MemoryPressureMB > 0 {
		s.ballast = allocateBallast(cfg.MemoryPressureMB)
	}

	return s
}

// allocateBallast allocates mb megabytes and touches one byte per page so
// the pages are resident rather than lazily-committed.
func allocateBallast(mb int) []byte {
	buf := make([]byte, mb*1024*1024)
	for i := 0; i < len(buf); i += pageSize {
		buf[i] = 1
	}
	return buf
}

// Handler returns the fully-wired http.Handler for the demo-service.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/health", s.instrument("/health", http.HandlerFunc(s.handleHealth)))
	mux.Handle("/version", s.instrument("/version", http.HandlerFunc(s.handleVersion)))
	mux.Handle("/work", s.instrument("/work", http.HandlerFunc(s.handleWork)))
	mux.Handle("/metrics", promhttp.HandlerFor(s.registry, promhttp.HandlerOpts{Registry: s.registry}))
	return mux
}
