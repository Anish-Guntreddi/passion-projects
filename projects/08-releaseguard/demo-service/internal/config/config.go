// Package config loads and validates the demo-service's runtime configuration.
//
// Every knob here exists so failure-injection scenarios (Phase 7 of the
// ReleaseGuard roadmap) can flip an environment variable and reproduce a
// specific canary failure mode -- latency regression, elevated error rate,
// dependency outage, or memory pressure -- without touching code. Phase 0/1
// only need the plumbing and validation; the scenario scripts land later.
package config

import (
	"fmt"
	"strconv"
	"strings"
)

// Track identifies which rollout identity a running instance represents.
// The controller (Phase 4+) will use this label to distinguish stable and
// canary telemetry for the same service.
type Track string

const (
	TrackStable Track = "stable"
	TrackCanary Track = "canary"
)

// Config holds every environment-derived setting the demo-service reads at
// startup. All fields are immutable for the lifetime of the process --
// changing behavior means restarting the pod, which mirrors how a real
// canary rollout swaps configuration via a new deployment revision.
type Config struct {
	// Port is the TCP port the HTTP server listens on.
	Port int

	// ReleaseTrack labels this instance as "stable" or "canary".
	ReleaseTrack Track

	// ReleaseVersion is the version string this instance reports on
	// /version and attaches to every metric it emits.
	ReleaseVersion string

	// BaseLatencyMS is the baseline simulated processing time for /work.
	BaseLatencyMS int

	// LatencyJitterMS is the maximum random jitter (uniformly distributed,
	// added on top of BaseLatencyMS) applied to /work.
	LatencyJitterMS int

	// ExtraLatencyMS is a constant, deterministic amount of extra latency
	// added to every /work call. Used to simulate a latency regression.
	ExtraLatencyMS int

	// ErrorRate is the probability in [0,1] that the simulated dependency
	// call inside /work fails, producing a 5xx response.
	ErrorRate float64

	// DependencyDown forces every /work call to fail regardless of
	// ErrorRate, simulating a hard dependency outage.
	DependencyDown bool

	// MemoryPressureMB is the amount of memory (in MiB) the process
	// allocates and retains at startup to simulate resource pressure.
	// The allocation is visible via the Go runtime metrics exposed on
	// /metrics (go_memstats_alloc_bytes etc.).
	MemoryPressureMB int

	// OTLPExporterEndpoint is the host:port of an OTLP/HTTP collector that
	// trace spans are exported to (e.g. "otel-collector.releaseguard.svc.
	// cluster.local:4318"). Empty disables trace export entirely -- the
	// service still creates spans internally (so handler code never has to
	// branch on whether tracing is "on"), it just uses a no-op provider
	// that discards them instead of touching the network. This is what
	// lets `go run`/`go test` behave identically with or without a
	// collector nearby, mirroring how every other fault-injection knob
	// here defaults to "off"/harmless.
	OTLPExporterEndpoint string

	// OTELTracesSampleRatio is the fraction (in [0,1]) of traces sampled
	// when tracing is enabled (OTLPExporterEndpoint is set). Ignored when
	// tracing is disabled. 1.0 (sample everything) is the default -- this
	// is a demo service used to generate small, deliberately-inspectable
	// failure scenarios, not a high-QPS production service, so there is no
	// cardinality/cost reason to sample less than everything.
	OTELTracesSampleRatio float64
}

// defaults returns a Config representing a healthy, stable instance.
func defaults() Config {
	return Config{
		Port:                  8080,
		ReleaseTrack:          TrackStable,
		ReleaseVersion:        "dev",
		BaseLatencyMS:         20,
		LatencyJitterMS:       10,
		ExtraLatencyMS:        0,
		ErrorRate:             0.0,
		DependencyDown:        false,
		MemoryPressureMB:      0,
		OTLPExporterEndpoint:  "",
		OTELTracesSampleRatio: 1.0,
	}
}

// Getenv matches os.Getenv's signature. Load takes it as a parameter (rather
// than calling os.Getenv directly) so tests can supply a fake environment
// without mutating global process state.
type Getenv func(key string) string

// Load reads and validates configuration from the supplied environment
// accessor, applying defaults() for any unset variable. It returns an error
// naming the first invalid value it encounters; the caller (main) should
// treat that as fatal rather than falling back to a guessed value -- silent
// misconfiguration of a canary is exactly the failure mode this project
// exists to prevent.
func Load(getenv Getenv) (Config, error) {
	cfg := defaults()

	if v := getenv("PORT"); v != "" {
		port, err := strconv.Atoi(v)
		if err != nil || port < 1 || port > 65535 {
			return Config{}, fmt.Errorf("config: PORT must be an integer in [1,65535], got %q", v)
		}
		cfg.Port = port
	}

	if v := getenv("RELEASE_TRACK"); v != "" {
		switch Track(strings.ToLower(v)) {
		case TrackStable:
			cfg.ReleaseTrack = TrackStable
		case TrackCanary:
			cfg.ReleaseTrack = TrackCanary
		default:
			return Config{}, fmt.Errorf("config: RELEASE_TRACK must be %q or %q, got %q", TrackStable, TrackCanary, v)
		}
	}

	if v := getenv("RELEASE_VERSION"); v != "" {
		cfg.ReleaseVersion = v
	}

	if v := getenv("BASE_LATENCY_MS"); v != "" {
		n, err := parseNonNegativeInt(v)
		if err != nil {
			return Config{}, fmt.Errorf("config: BASE_LATENCY_MS %w", err)
		}
		cfg.BaseLatencyMS = n
	}

	if v := getenv("LATENCY_JITTER_MS"); v != "" {
		n, err := parseNonNegativeInt(v)
		if err != nil {
			return Config{}, fmt.Errorf("config: LATENCY_JITTER_MS %w", err)
		}
		cfg.LatencyJitterMS = n
	}

	if v := getenv("EXTRA_LATENCY_MS"); v != "" {
		n, err := parseNonNegativeInt(v)
		if err != nil {
			return Config{}, fmt.Errorf("config: EXTRA_LATENCY_MS %w", err)
		}
		cfg.ExtraLatencyMS = n
	}

	if v := getenv("ERROR_RATE"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 || f > 1 {
			return Config{}, fmt.Errorf("config: ERROR_RATE must be a float in [0,1], got %q", v)
		}
		cfg.ErrorRate = f
	}

	if v := getenv("DEPENDENCY_DOWN"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf("config: DEPENDENCY_DOWN must be a boolean, got %q", v)
		}
		cfg.DependencyDown = b
	}

	if v := getenv("MEMORY_PRESSURE_MB"); v != "" {
		n, err := parseNonNegativeInt(v)
		if err != nil {
			return Config{}, fmt.Errorf("config: MEMORY_PRESSURE_MB %w", err)
		}
		cfg.MemoryPressureMB = n
	}

	if v := getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); v != "" {
		cfg.OTLPExporterEndpoint = v
	}

	if v := getenv("OTEL_TRACES_SAMPLE_RATIO"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 || f > 1 {
			return Config{}, fmt.Errorf("config: OTEL_TRACES_SAMPLE_RATIO must be a float in [0,1], got %q", v)
		}
		cfg.OTELTracesSampleRatio = f
	}

	return cfg, nil
}

func parseNonNegativeInt(v string) (int, error) {
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("must be an integer, got %q", v)
	}
	if n < 0 {
		return 0, fmt.Errorf("must be >= 0, got %d", n)
	}
	return n, nil
}
