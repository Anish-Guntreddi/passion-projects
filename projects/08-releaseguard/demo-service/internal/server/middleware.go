package server

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// statusRecorder wraps an http.ResponseWriter to capture the status code a
// handler wrote, defaulting to 200 to match net/http's own behavior when a
// handler never calls WriteHeader explicitly (the first Write() implies
// 200 OK).
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// instrument wraps next so every request through it updates
// http_requests_total and http_request_duration_seconds with route,
// method, status, release version and track labels. It is the single
// place request-level metrics are recorded, so every handler gets
// consistent instrumentation for free.
func (s *Server) instrument(route string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		start := time.Now()
		next.ServeHTTP(rec, r)
		duration := time.Since(start).Seconds()

		version := s.cfg.ReleaseVersion
		track := string(s.cfg.ReleaseTrack)

		s.metrics.RequestDuration.With(prometheus.Labels{
			"route":   route,
			"method":  r.Method,
			"version": version,
			"track":   track,
		}).Observe(duration)

		s.metrics.RequestsTotal.WithLabelValues(
			route,
			r.Method,
			strconv.Itoa(rec.status),
			version,
			track,
		).Inc()
	})
}
