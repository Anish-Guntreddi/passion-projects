# Test organization

This directory holds cross-component and shell-driven test suites, per the
spec's repo structure (`tests/{unit,integration,e2e,failure}/`). Go unit
tests for a given package live beside that package's source
(`*_test.go`, Go's standard convention) rather than under `tests/unit/`,
because `go test` needs them there to access package-private state -- see
e.g. `demo-service/internal/config/config_test.go`,
`demo-service/internal/server/handlers_test.go`. `tests/unit/` exists as a
placeholder for future non-Go or cross-package unit-style suites and is
currently empty.

| Directory | Populated by | Status |
|---|---|---|
| `unit/` | (reserved; Go unit tests are co-located with source instead) | empty |
| `integration/` | Phase 2+ (Prometheus query adapter), Phase 6 (Kubernetes adapter with fake client) | empty |
| `e2e/` | Phase 1 (this change): `smoke_test.sh` | **populated** |
| `failure/` | Phase 7 (scripted latency/error/missing-telemetry/saturation injection) | empty |

Run `make test` (from the project root) for the Go unit test suite, and
`make cluster-up` / `make smoke-test` for the Phase 1 e2e check.
