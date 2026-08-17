package policy_test

import (
	"encoding/json"
	"testing"
	"time"

	"releaseguard/controller/internal/policy"
)

func TestDuration_JSONRoundTrip(t *testing.T) {
	d := policy.MustParseDuration("1m30s")
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	if string(b) != `"1m30s"` {
		t.Errorf("Marshal = %s, want %q", b, `"1m30s"`)
	}

	var got policy.Duration
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if got.Duration != 90*time.Second {
		t.Errorf("round-tripped Duration = %v, want 90s", got.Duration)
	}
}

func TestDuration_UnmarshalNumeric(t *testing.T) {
	var got policy.Duration
	if err := json.Unmarshal([]byte(`5000000000`), &got); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if got.Duration != 5*time.Second {
		t.Errorf("Duration = %v, want 5s", got.Duration)
	}
}

func TestDuration_UnmarshalInvalid(t *testing.T) {
	var got policy.Duration
	if err := json.Unmarshal([]byte(`"not-a-duration"`), &got); err == nil {
		t.Fatal("Unmarshal(invalid duration string) returned nil error")
	}
}

func TestParseDuration_Invalid(t *testing.T) {
	if _, err := policy.ParseDuration("banana"); err == nil {
		t.Fatal("ParseDuration(\"banana\") returned nil error")
	}
}

func TestMustParseDuration_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("MustParseDuration did not panic on an invalid duration")
		}
	}()
	policy.MustParseDuration("banana")
}
