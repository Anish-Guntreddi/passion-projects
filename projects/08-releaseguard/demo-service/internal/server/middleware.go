package server

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
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
// method, status, release version and track labels, AND opens an OTel
// span for the request. It is the single place request-level telemetry
// (both the Prometheus metrics and the trace span) is recorded, so every
// handler gets consistent instrumentation for free and the two signal
// types can never drift apart on what a "request" means.
func (s *Server) instrument(route string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		version := s.cfg.ReleaseVersion
		track := string(s.cfg.ReleaseTrack)

		ctx, span := s.tracer.Start(r.Context(), route, trace.WithAttributes(
			attribute.String("http.method", r.Method),
			attribute.String("http.route", route),
			attribute.String("release.track", track),
			attribute.String("release.version", version),
		))
		defer span.End()
		r = r.WithContext(ctx)

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		start := time.Now()
		next.ServeHTTP(rec, r)
		duration := time.Since(start).Seconds()

		span.SetAttributes(attribute.Int("http.status_code", rec.status))
		if rec.status >= http.StatusInternalServerError {
			span.SetStatus(codes.Error, fmt.Sprintf("http status %d", rec.status))
		}

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
