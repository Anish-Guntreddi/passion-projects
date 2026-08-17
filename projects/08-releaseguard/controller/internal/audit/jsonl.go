package audit

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// JSONLLogger is the production Logger (ADR 0005): one JSON object per
// line, appended to a file. Append-only writes plus an fsync after every
// record is what makes this safe to reason about across a controller
// restart mid-rollout (§1.7's reliability requirement) -- a torn write can
// at worst leave one incomplete trailing line, never corrupt a prior
// record, and ReadJSONL below tolerates exactly that.
type JSONLLogger struct {
	mu sync.Mutex
	f  *os.File
}

// NewJSONLLogger opens (creating if needed, including parent directories)
// the file at path for appending and returns a ready-to-use JSONLLogger.
// Safe to call again on an existing file -- new events are appended after
// whatever is already there, never overwriting prior history.
func NewJSONLLogger(path string) (*JSONLLogger, error) {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("audit: create directory %s: %w", dir, err)
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("audit: open %s: %w", path, err)
	}
	return &JSONLLogger{f: f}, nil
}

// Record appends e as one JSON line and fsyncs before returning, so a
// successful Record means the event has actually reached disk, not just
// the OS page cache.
func (l *JSONLLogger) Record(e Event) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("audit: marshal event: %w", err)
	}
	b = append(b, '\n')
	if _, err := l.f.Write(b); err != nil {
		return fmt.Errorf("audit: write event: %w", err)
	}
	if err := l.f.Sync(); err != nil {
		return fmt.Errorf("audit: fsync: %w", err)
	}
	return nil
}

// Close closes the underlying file. Safe to call once; ReleaseGuard's
// controller process holds one JSONLLogger open for its lifetime in
// practice, but tests and short-lived CLI commands (cmd/controller
// simulate) want a clean Close.
func (l *JSONLLogger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.f.Close()
}

// ReadJSONL reads every event from a JSONL audit file, in order. A blank
// trailing line (e.g. the file ends in "\n" with nothing after it, the
// normal case) is skipped; a genuinely truncated trailing line (a crash
// mid-write, before Record's fsync completed) is also skipped rather than
// erroring the whole read, since everything before it is still valid,
// complete history -- exactly the crash-safety property JSONL's
// append-only design is chosen for (ADR 0005).
func ReadJSONL(path string) ([]Event, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("audit: open %s: %w", path, err)
	}
	defer f.Close()

	var events []Event
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var e Event
		if err := json.Unmarshal(line, &e); err != nil {
			continue // tolerate a truncated trailing line; see doc comment
		}
		events = append(events, e)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("audit: read %s: %w", path, err)
	}
	return events, nil
}
