package policy

import (
	"encoding/json"
	"fmt"
	"time"
)

// Duration wraps time.Duration with JSON (and therefore, via
// sigs.k8s.io/yaml, YAML) marshaling as a Go duration string ("30s",
// "5m") rather than a raw integer nanosecond count -- policy files are
// meant to be hand-written and hand-reviewed (spec: "policy threshold
// values are human-review decisions"), and nobody hand-reviews nanosecond
// counts.
type Duration struct {
	time.Duration
}

// ParseDuration is a small convenience constructor, mainly for tests and
// for building policies programmatically (e.g. the CLI's default-policy
// scaffold) without spelling out the struct literal.
func ParseDuration(s string) (Duration, error) {
	d, err := time.ParseDuration(s)
	if err != nil {
		return Duration{}, fmt.Errorf("policy: invalid duration %q: %w", s, err)
	}
	return Duration{d}, nil
}

// MustParseDuration panics on an invalid duration string. Reserved for
// package-level constants and tests where the input is a literal known
// good at compile time.
func MustParseDuration(s string) Duration {
	d, err := ParseDuration(s)
	if err != nil {
		panic(err)
	}
	return d
}

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.Duration.String())
}

// UnmarshalJSON accepts either a Go duration string ("30s") or a plain
// number, interpreted as nanoseconds for symmetry with time.Duration's own
// underlying representation. String form is expected in practice; numeric
// form is accepted so a policy generated programmatically (e.g. by a
// future tool) is not forced to stringify durations itself.
func (d *Duration) UnmarshalJSON(data []byte) error {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("policy: invalid duration: %w", err)
	}
	switch v := raw.(type) {
	case string:
		parsed, err := time.ParseDuration(v)
		if err != nil {
			return fmt.Errorf("policy: invalid duration %q: %w", v, err)
		}
		d.Duration = parsed
	case float64:
		d.Duration = time.Duration(v)
	case nil:
		d.Duration = 0
	default:
		return fmt.Errorf("policy: invalid duration value %v (%T)", v, v)
	}
	return nil
}
