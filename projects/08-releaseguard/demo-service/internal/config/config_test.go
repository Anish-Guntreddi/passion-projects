package config

import (
	"math"
	"testing"
)

// fakeEnv builds a Getenv closure over a fixed map, giving each test a
// hermetic environment instead of mutating process-wide state via
// os.Setenv (which would make tests order-dependent and unsafe to
// parallelize).
func fakeEnv(vars map[string]string) Getenv {
	return func(key string) string { return vars[key] }
}

func TestLoad_Defaults(t *testing.T) {
	cfg, err := Load(fakeEnv(nil))
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	want := Config{
		Port:             8080,
		ReleaseTrack:     TrackStable,
		ReleaseVersion:   "dev",
		BaseLatencyMS:    20,
		LatencyJitterMS:  10,
		ExtraLatencyMS:   0,
		ErrorRate:        0.0,
		DependencyDown:   false,
		MemoryPressureMB: 0,
	}
	if cfg != want {
		t.Fatalf("Load() defaults = %+v, want %+v", cfg, want)
	}
}

func TestLoad_ValidOverrides(t *testing.T) {
	cfg, err := Load(fakeEnv(map[string]string{
		"PORT":               "9090",
		"RELEASE_TRACK":      "canary",
		"RELEASE_VERSION":    "v1.2.3",
		"BASE_LATENCY_MS":    "50",
		"LATENCY_JITTER_MS":  "5",
		"EXTRA_LATENCY_MS":   "500",
		"ERROR_RATE":         "0.25",
		"DEPENDENCY_DOWN":    "true",
		"MEMORY_PRESSURE_MB": "16",
	}))
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.Port != 9090 {
		t.Errorf("Port = %d, want 9090", cfg.Port)
	}
	if cfg.ReleaseTrack != TrackCanary {
		t.Errorf("ReleaseTrack = %q, want %q", cfg.ReleaseTrack, TrackCanary)
	}
	if cfg.ReleaseVersion != "v1.2.3" {
		t.Errorf("ReleaseVersion = %q, want v1.2.3", cfg.ReleaseVersion)
	}
	if cfg.BaseLatencyMS != 50 {
		t.Errorf("BaseLatencyMS = %d, want 50", cfg.BaseLatencyMS)
	}
	if cfg.LatencyJitterMS != 5 {
		t.Errorf("LatencyJitterMS = %d, want 5", cfg.LatencyJitterMS)
	}
	if cfg.ExtraLatencyMS != 500 {
		t.Errorf("ExtraLatencyMS = %d, want 500", cfg.ExtraLatencyMS)
	}
	if math.Abs(cfg.ErrorRate-0.25) > 1e-9 {
		t.Errorf("ErrorRate = %v, want 0.25", cfg.ErrorRate)
	}
	if !cfg.DependencyDown {
		t.Errorf("DependencyDown = false, want true")
	}
	if cfg.MemoryPressureMB != 16 {
		t.Errorf("MemoryPressureMB = %d, want 16", cfg.MemoryPressureMB)
	}
}

func TestLoad_ReleaseTrackCaseInsensitive(t *testing.T) {
	cfg, err := Load(fakeEnv(map[string]string{"RELEASE_TRACK": "CANARY"}))
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}
	if cfg.ReleaseTrack != TrackCanary {
		t.Errorf("ReleaseTrack = %q, want %q", cfg.ReleaseTrack, TrackCanary)
	}
}

func TestLoad_InvalidValues(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
	}{
		{"bad port not a number", map[string]string{"PORT": "abc"}},
		{"port zero", map[string]string{"PORT": "0"}},
		{"port too large", map[string]string{"PORT": "70000"}},
		{"bad release track", map[string]string{"RELEASE_TRACK": "primary"}},
		{"negative base latency", map[string]string{"BASE_LATENCY_MS": "-1"}},
		{"non-numeric base latency", map[string]string{"BASE_LATENCY_MS": "fast"}},
		{"negative jitter", map[string]string{"LATENCY_JITTER_MS": "-5"}},
		{"negative extra latency", map[string]string{"EXTRA_LATENCY_MS": "-100"}},
		{"error rate below zero", map[string]string{"ERROR_RATE": "-0.1"}},
		{"error rate above one", map[string]string{"ERROR_RATE": "1.1"}},
		{"error rate not a number", map[string]string{"ERROR_RATE": "high"}},
		{"dependency down not a bool", map[string]string{"DEPENDENCY_DOWN": "maybe"}},
		{"negative memory pressure", map[string]string{"MEMORY_PRESSURE_MB": "-1"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(fakeEnv(tc.env))
			if err == nil {
				t.Fatalf("Load() with %v: expected error, got nil", tc.env)
			}
		})
	}
}

func TestLoad_EmptyStringLeavesDefault(t *testing.T) {
	// An explicitly-set-but-empty env var should behave like "unset" rather
	// than being parsed (and rejected) as an empty value. This matters in
	// Kubernetes where an env var can be defined with an empty valueFrom.
	cfg, err := Load(fakeEnv(map[string]string{"PORT": ""}))
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}
	if cfg.Port != 8080 {
		t.Errorf("Port = %d, want default 8080", cfg.Port)
	}
}
