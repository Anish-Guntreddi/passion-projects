package policy

import (
	"fmt"
	"os"

	"sigs.k8s.io/yaml"
)

// Load reads and parses the policy document at path. Both YAML and JSON
// are accepted regardless of file extension (see Parse).
func Load(path string) (*Policy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("policy: read %s: %w", path, err)
	}
	p, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("policy: %s: %w", path, err)
	}
	return p, nil
}

// Parse unmarshals a policy document (FR4: "Typed policy config (YAML/JSON
// with validation)"). sigs.k8s.io/yaml converts YAML to JSON internally
// and then decodes via encoding/json, so a single set of `json:` struct
// tags (types.go) drives both formats with no separate YAML tag set to
// keep in sync, and JSON input (which is already valid YAML) works
// unmodified through the same code path. Defaults are applied before
// validation, so e.g. an unset OnMissingTelemetry is validated as its
// default (OnMissingPause), not rejected as empty.
func Parse(data []byte) (*Policy, error) {
	var p Policy
	if err := yaml.UnmarshalStrict(data, &p); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	p.ApplyDefaults()
	if err := p.Validate(); err != nil {
		return nil, fmt.Errorf("invalid policy: %w", err)
	}
	return &p, nil
}

// ApplyDefaults fills in the policy's safe defaults for fields left unset.
// Every default chosen here is the *safer* option where the spec allows a
// choice (e.g. OnMissingPause, never a promote-equivalent action) --
// see docs/decisions/0010-missing-telemetry-and-pause-semantics.md.
func (p *Policy) ApplyDefaults() {
	if p.Evaluation.OnMissingTelemetry == "" {
		p.Evaluation.OnMissingTelemetry = OnMissingPause
	}
	if p.Evaluation.BaselineTrack == "" {
		p.Evaluation.BaselineTrack = "stable"
	}
}
