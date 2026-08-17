# Phase 0 test coverage: raw, measured `go test` output

**What this file is:** the exact, unedited output of `go test ./... -race
-cover -v` for the `demo-service` module, committed so the coverage
percentages quoted in `README.md`'s Testing section are backed by a
reproducible artifact instead of numbers typed by hand into prose (which
drift silently as tests are added/removed). Re-run the command below to
regenerate this file whenever the test suite changes; do not hand-edit the
percentages in either file without also updating the other.

## Environment

| | |
|---|---|
| Date (UTC) | 2026-08-17 |
| Host | Windows 11 Pro, WSL2 Ubuntu 24.04.1 LTS |
| Go | go1.22.2 linux/amd64 |
| Repo commit at run time | working tree post-fix (see git log for the commit this artifact ships in) |

## Command

```bash
cd demo-service
go vet ./...
go test ./... -race -cover -v
```

## Measured result

```
	releaseguard/demo-service/cmd/demo-service		coverage: 0.0% of statements
=== RUN   TestLoad_Defaults
--- PASS: TestLoad_Defaults (0.00s)
=== RUN   TestLoad_ValidOverrides
--- PASS: TestLoad_ValidOverrides (0.00s)
=== RUN   TestLoad_ReleaseTrackCaseInsensitive
--- PASS: TestLoad_ReleaseTrackCaseInsensitive (0.00s)
=== RUN   TestLoad_InvalidValues
=== RUN   TestLoad_InvalidValues/bad_port_not_a_number
=== RUN   TestLoad_InvalidValues/port_zero
=== RUN   TestLoad_InvalidValues/port_too_large
=== RUN   TestLoad_InvalidValues/bad_release_track
=== RUN   TestLoad_InvalidValues/negative_base_latency
=== RUN   TestLoad_InvalidValues/non-numeric_base_latency
=== RUN   TestLoad_InvalidValues/negative_jitter
=== RUN   TestLoad_InvalidValues/negative_extra_latency
=== RUN   TestLoad_InvalidValues/error_rate_below_zero
=== RUN   TestLoad_InvalidValues/error_rate_above_one
=== RUN   TestLoad_InvalidValues/error_rate_not_a_number
=== RUN   TestLoad_InvalidValues/dependency_down_not_a_bool
=== RUN   TestLoad_InvalidValues/negative_memory_pressure
--- PASS: TestLoad_InvalidValues (0.00s)
    --- PASS: TestLoad_InvalidValues/bad_port_not_a_number (0.00s)
    --- PASS: TestLoad_InvalidValues/port_zero (0.00s)
    --- PASS: TestLoad_InvalidValues/port_too_large (0.00s)
    --- PASS: TestLoad_InvalidValues/bad_release_track (0.00s)
    --- PASS: TestLoad_InvalidValues/negative_base_latency (0.00s)
    --- PASS: TestLoad_InvalidValues/non-numeric_base_latency (0.00s)
    --- PASS: TestLoad_InvalidValues/negative_jitter (0.00s)
    --- PASS: TestLoad_InvalidValues/negative_extra_latency (0.00s)
    --- PASS: TestLoad_InvalidValues/error_rate_below_zero (0.00s)
    --- PASS: TestLoad_InvalidValues/error_rate_above_one (0.00s)
    --- PASS: TestLoad_InvalidValues/error_rate_not_a_number (0.00s)
    --- PASS: TestLoad_InvalidValues/dependency_down_not_a_bool (0.00s)
    --- PASS: TestLoad_InvalidValues/negative_memory_pressure (0.00s)
=== RUN   TestLoad_EmptyStringLeavesDefault
--- PASS: TestLoad_EmptyStringLeavesDefault (0.00s)
PASS
coverage: 98.0% of statements
ok  	releaseguard/demo-service/internal/config	(cached)	coverage: 98.0% of statements
	releaseguard/demo-service/internal/version		coverage: 0.0% of statements
=== RUN   TestCall_Success
--- PASS: TestCall_Success (0.00s)
=== RUN   TestCall_TransientFailure
--- PASS: TestCall_TransientFailure (0.00s)
=== RUN   TestCall_BoundaryIsSuccess
--- PASS: TestCall_BoundaryIsSuccess (0.00s)
=== RUN   TestCall_ZeroErrorRateNeverFails
--- PASS: TestCall_ZeroErrorRateNeverFails (0.00s)
=== RUN   TestCall_ForcedDownOverridesErrorRate
--- PASS: TestCall_ForcedDownOverridesErrorRate (0.00s)
=== RUN   TestCall_ConcurrentUseIsRaceFree
--- PASS: TestCall_ConcurrentUseIsRaceFree (0.00s)
=== RUN   TestNew_DeterministicWithSameSeed
--- PASS: TestNew_DeterministicWithSameSeed (0.00s)
PASS
coverage: 100.0% of statements
ok  	releaseguard/demo-service/internal/dependency	1.055s	coverage: 100.0% of statements
=== RUN   TestHandleHealth_OK
--- PASS: TestHandleHealth_OK (0.00s)
=== RUN   TestHandleHealth_ReportsDegradedDependencyButStays200
--- PASS: TestHandleHealth_ReportsDegradedDependencyButStays200 (0.00s)
=== RUN   TestHandleHealth_ReportsDownDependencyButStays200
--- PASS: TestHandleHealth_ReportsDownDependencyButStays200 (0.00s)
=== RUN   TestHandleVersion
--- PASS: TestHandleVersion (0.00s)
=== RUN   TestHandleWork_SucceedsWhenErrorRateZero
--- PASS: TestHandleWork_SucceedsWhenErrorRateZero (0.00s)
=== RUN   TestHandleWork_FailsWhenErrorRateOne
--- PASS: TestHandleWork_FailsWhenErrorRateOne (0.00s)
=== RUN   TestHandleWork_FailsWhenDependencyDown
--- PASS: TestHandleWork_FailsWhenDependencyDown (0.00s)
=== RUN   TestHandleWork_RespectsExtraLatency
--- PASS: TestHandleWork_RespectsExtraLatency (0.03s)
=== RUN   TestHandleWork_CancelledContextAbortsWithoutPanic
--- PASS: TestHandleWork_CancelledContextAbortsWithoutPanic (0.00s)
=== RUN   TestMetricsEndpoint_ExposesKnownMetricNames
--- PASS: TestMetricsEndpoint_ExposesKnownMetricNames (0.00s)
=== RUN   TestInstrument_RecordsRequestsTotal
--- PASS: TestInstrument_RecordsRequestsTotal (0.00s)
=== RUN   TestInstrument_RecordsErrorStatusLabel
--- PASS: TestInstrument_RecordsErrorStatusLabel (0.00s)
=== RUN   TestInstrument_RecordsDependencyFailureReason
--- PASS: TestInstrument_RecordsDependencyFailureReason (0.00s)
=== RUN   TestInstrument_RequestDurationObserved
--- PASS: TestInstrument_RequestDurationObserved (0.00s)
PASS
coverage: 90.0% of statements
ok  	releaseguard/demo-service/internal/server	1.077s	coverage: 90.0% of statements
```

`cmd/demo-service` (0.0%) and `internal/version` (0.0%) are the entrypoint
`main()` and a struct of ldflags-injected string vars respectively --
there is no branching logic in either to unit test; both are exercised
end-to-end by `deploy/local-cluster/setup-cluster.sh` +
`tests/e2e/smoke_test.sh` instead (the `/version` endpoint asserts on
`internal/version`'s output, and the binary itself only runs at all if
`cmd/demo-service`'s `main()` works).

## Interpretation

- `internal/config` 98.0%, `internal/dependency` 100.0%, `internal/server`
  90.0% -- these are the numbers `README.md` quotes; this file is the
  artifact backing them.
- `TestCall_ConcurrentUseIsRaceFree` (added alongside the fix for the
  shared-Mock-RNG data race -- see `internal/dependency/mock.go` and its
  test file) is included in this run and passes under `-race`.
