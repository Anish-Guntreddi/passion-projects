package audit_test

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"releaseguard/controller/internal/audit"
)

func TestJSONLLogger_WriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")

	l, err := audit.NewJSONLLogger(path)
	if err != nil {
		t.Fatalf("NewJSONLLogger: %v", err)
	}

	events := []audit.Event{
		{RolloutID: "r1", Timestamp: "2026-01-01T00:00:00Z", EventType: "state_transition", FromState: "PENDING", ToState: "DEPLOYING"},
		{RolloutID: "r1", Timestamp: "2026-01-01T00:00:15Z", EventType: "state_transition", FromState: "DEPLOYING", ToState: "OBSERVING", Reasons: []string{"CANARY_DEPLOYED"}},
		{RolloutID: "r1", Timestamp: "2026-01-01T00:00:45Z", EventType: "state_transition", FromState: "OBSERVING", ToState: "PROMOTING", Decision: "PROMOTE", Reasons: []string{"CONSECUTIVE_HEALTHY_WINDOWS_MET"}},
	}
	for _, e := range events {
		if err := l.Record(e); err != nil {
			t.Fatalf("Record: %v", err)
		}
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, err := audit.ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	if len(got) != len(events) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(events))
	}
	for i := range events {
		if !reflect.DeepEqual(got[i], events[i]) {
			t.Errorf("event[%d] = %+v, want %+v", i, got[i], events[i])
		}
	}
}

func TestJSONLLogger_AppendsAcrossReopen(t *testing.T) {
	// Simulates a controller restart: a fresh JSONLLogger opened against
	// an existing audit file must append after prior history, never
	// truncate or overwrite it (§1.7: "controller restart during
	// observation/action").
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")

	l1, err := audit.NewJSONLLogger(path)
	if err != nil {
		t.Fatalf("NewJSONLLogger (1st): %v", err)
	}
	if err := l1.Record(audit.Event{RolloutID: "r1", ToState: "PENDING"}); err != nil {
		t.Fatalf("Record (1st): %v", err)
	}
	if err := l1.Close(); err != nil {
		t.Fatalf("Close (1st): %v", err)
	}

	l2, err := audit.NewJSONLLogger(path) // simulated restart
	if err != nil {
		t.Fatalf("NewJSONLLogger (2nd): %v", err)
	}
	if err := l2.Record(audit.Event{RolloutID: "r1", ToState: "DEPLOYING"}); err != nil {
		t.Fatalf("Record (2nd): %v", err)
	}
	if err := l2.Close(); err != nil {
		t.Fatalf("Close (2nd): %v", err)
	}

	got, err := audit.ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2 (one from each logger instance)", len(got))
	}
	if got[0].ToState != "PENDING" || got[1].ToState != "DEPLOYING" {
		t.Errorf("got = %+v, want [PENDING, DEPLOYING] in order", got)
	}
}

func TestJSONLLogger_CreatesParentDirectory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deeper", "audit.jsonl")

	l, err := audit.NewJSONLLogger(path)
	if err != nil {
		t.Fatalf("NewJSONLLogger: %v", err)
	}
	defer l.Close()
	if err := l.Record(audit.Event{RolloutID: "r1"}); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("audit file not created at %s: %v", path, err)
	}
}

func TestJSONLLogger_ConcurrentRecordIsSafe(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")
	l, err := audit.NewJSONLLogger(path)
	if err != nil {
		t.Fatalf("NewJSONLLogger: %v", err)
	}
	defer l.Close()

	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			_ = l.Record(audit.Event{RolloutID: "r1", ToState: fmt.Sprintf("STATE_%d", i)})
		}(i)
	}
	wg.Wait()

	got, err := audit.ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	if len(got) != n {
		t.Fatalf("len(got) = %d, want %d (concurrent writes must not be lost or corrupted)", len(got), n)
	}
}

func TestReadJSONL_ToleratesTruncatedTrailingLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")
	content := `{"rollout_id":"r1","to_state":"PENDING"}
{"rollout_id":"r1","to_state":"DEPL`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := audit.ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(got) = %d, want 1 (complete record kept, truncated one skipped)", len(got))
	}
	if got[0].ToState != "PENDING" {
		t.Errorf("got[0].ToState = %q, want PENDING", got[0].ToState)
	}
}

func TestReadJSONL_MissingFile(t *testing.T) {
	if _, err := audit.ReadJSONL(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Fatal("ReadJSONL(missing file) returned nil error")
	}
}

func TestInMemoryLogger(t *testing.T) {
	l := audit.NewInMemoryLogger()
	if err := l.Record(audit.Event{RolloutID: "r1", ToState: "PENDING"}); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if err := l.Record(audit.Event{RolloutID: "r1", ToState: "DEPLOYING"}); err != nil {
		t.Fatalf("Record: %v", err)
	}
	events := l.Events()
	if len(events) != 2 {
		t.Fatalf("len(Events()) = %d, want 2", len(events))
	}

	// Returned slice must be a copy, not a view into internal state.
	events[0].ToState = "MUTATED"
	if l.Events()[0].ToState != "PENDING" {
		t.Error("mutating the returned slice affected the logger's internal state")
	}
}
