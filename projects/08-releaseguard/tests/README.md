# Test organization

This directory holds cross-component and shell-driven test suites, per the
spec's repo structure (`tests/{unit,integration,e2e,failure}/`). Go unit
tests for a given package live beside that package's source
(`*_test.go`, Go's standard convention) rather than under `tests/unit/`,
because `go test` needs them there to access package-private state -- see
e.g. `demo-service/internal/config/config_test.go`,
`demo-service/internal/server/handlers_test.go`, and (Phase 4-5)
`controller/internal/policy/policy_test.go`,
`controller/internal/rollout/machine_test.go`, etc. `tests/unit/` exists
as a placeholder for future non-Go or cross-package unit-style suites and
is currently empty.

| Directory | Populated by | Status |
|---|---|---|
| `unit/` | (reserved; Go unit tests are co-located with source instead) | empty |
| `integration/` | Phase 4 (Prometheus query adapter), Phase 6 (Kubernetes adapter with fake client) | empty -- see note below |
| `e2e/` | Phase 1: `smoke_test.sh`; Phase 3: `canary_smoke_test.sh` | **populated** |
| `failure/` | Phase 7 (scripted latency/error/missing-telemetry/saturation injection) | empty |

**Why `integration/` stays empty for Phase 4's Prometheus adapter test**:
same reason unit tests are co-located above, one level stronger. Go's
`internal/` package visibility rule is keyed off *import path*, not
filesystem nesting: `releaseguard/controller/internal/metrics` can only be
imported by code whose own import path is prefixed
`releaseguard/controller/`. `tests/` at the project root is not part of
the `releaseguard/controller` module (`controller/go.mod` is a separate
module from `demo-service/go.mod`, and neither includes `tests/`), so a
test file placed under `tests/integration/` could never import
`internal/metrics` regardless of how deeply it's nested on disk. The
httptest-backed Prometheus adapter integration test therefore lives at
`controller/internal/metrics/prometheus_test.go` instead -- it exercises
the real HTTP client end to end against a fake server speaking the actual
Prometheus API response shape, which is the substance `tests/integration/`
was reserved for; only the directory differs from what the repo structure
sketch in the spec implies. Phase 6's Kubernetes-adapter test is expected
to face the identical constraint and land the same way, inside
`controller/internal/rollout/` (or a future `internal/k8s/`) next to the
adapter it tests.

Run `make test` (from the project root) for the Go unit test suite
(demo-service **and** controller, Phase 4-5 onward), `make cluster-up` to
deploy both tracks and run both e2e smoke tests, `make smoke-test` /
`make smoke-test-canary` individually against whatever cluster is already
current, or `make controller-test` for just the controller module.
