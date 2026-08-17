package server

import (
	"encoding/json"
	"errors"
	"math/rand"
	"net/http"
	"time"

	"releaseguard/demo-service/internal/dependency"
	"releaseguard/demo-service/internal/version"
)

// healthResponse is the JSON body returned by GET /health.
//
// Design decision: /health always answers 200 as long as the process is
// alive, even when DependencyDown or a nonzero ErrorRate is configured.
// Kubernetes liveness/readiness probes hit this endpoint, and restarting
// or de-registering a pod because of an *intentionally injected* SLO
// failure would (a) defeat the failure-injection scenarios this project
// exists to run and (b) conflate two different concerns: "is the process
// alive" (kubelet's job) versus "is this release healthy" (ReleaseGuard's
// job, starting Phase 4). The Dependency field still reports the
// degraded/down state so it is observable, just not fatal to the probe.
type healthResponse struct {
	Status        string  `json:"status"`
	Track         string  `json:"track"`
	Version       string  `json:"version"`
	Dependency    string  `json:"dependency"`
	UptimeSeconds float64 `json:"uptime_seconds"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	depStatus := "ok"
	switch {
	case s.cfg.DependencyDown:
		depStatus = "down"
	case s.cfg.ErrorRate > 0:
		depStatus = "degraded"
	}

	writeJSON(w, http.StatusOK, healthResponse{
		Status:        "ok",
		Track:         string(s.cfg.ReleaseTrack),
		Version:       s.cfg.ReleaseVersion,
		Dependency:    depStatus,
		UptimeSeconds: time.Since(s.startedAt).Seconds(),
	})
}

// versionResponse is the JSON body returned by GET /version.
//
// ReleaseVersion and ImageVersion are deliberately separate fields:
// ImageVersion/GitCommit/BuildTime are baked into the binary at build time
// (via -ldflags, see internal/version) and identify *what code is
// running*. ReleaseVersion comes from the RELEASE_VERSION environment
// variable set by the Kubernetes deployment manifest and identifies *what
// rollout this pod represents* -- the value the controller and Prometheus
// labels key on. They usually match, but keeping them distinct lets the
// same immutable image be reused across a version bump driven purely by
// deployment metadata, which is a real deployment pattern worth being
// honest about here.
type versionResponse struct {
	ReleaseVersion string `json:"release_version"`
	ImageVersion   string `json:"image_version"`
	GitCommit      string `json:"git_commit"`
	BuildTime      string `json:"build_time"`
	Track          string `json:"track"`
}

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	info := version.Current()
	writeJSON(w, http.StatusOK, versionResponse{
		ReleaseVersion: s.cfg.ReleaseVersion,
		ImageVersion:   info.Version,
		GitCommit:      info.GitCommit,
		BuildTime:      info.BuildTime,
		Track:          string(s.cfg.ReleaseTrack),
	})
}

// workResponse is the JSON body returned by a successful GET /work.
type workResponse struct {
	Result    string `json:"result"`
	LatencyMS int64  `json:"latency_ms"`
	Track     string `json:"track"`
	Version   string `json:"version"`
}

// workErrorResponse is the JSON body returned when the simulated
// dependency call fails.
type workErrorResponse struct {
	Error  string `json:"error"`
	Reason string `json:"reason"`
}

// handleWork is the demo-service's main simulated workload: it sleeps for
// a configurable duration (baseline + jitter + injected extra latency) and
// then calls the mock dependency, which fails according to ErrorRate or
// DependencyDown. This is the single endpoint every failure-injection
// scenario in Phase 7 targets.
func (s *Server) handleWork(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	delayMS := s.cfg.BaseLatencyMS + s.cfg.ExtraLatencyMS
	if s.cfg.LatencyJitterMS > 0 {
		delayMS += rand.Intn(s.cfg.LatencyJitterMS + 1)
	}
	delay := time.Duration(delayMS) * time.Millisecond

	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
	}

	err := s.dep.Call(s.cfg.ErrorRate, s.cfg.DependencyDown)
	elapsed := time.Since(start)

	if err != nil {
		reason := "transient"
		if errors.Is(err, dependency.ErrDown) {
			reason = "down"
		}
		s.metrics.DependencyFailuresTotal.WithLabelValues(reason, s.cfg.ReleaseVersion, string(s.cfg.ReleaseTrack)).Inc()

		writeJSON(w, http.StatusServiceUnavailable, workErrorResponse{
			Error:  "dependency call failed",
			Reason: reason,
		})
		return
	}

	writeJSON(w, http.StatusOK, workResponse{
		Result:    "ok",
		LatencyMS: elapsed.Milliseconds(),
		Track:     string(s.cfg.ReleaseTrack),
		Version:   s.cfg.ReleaseVersion,
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
