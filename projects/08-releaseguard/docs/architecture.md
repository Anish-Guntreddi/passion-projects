# ReleaseGuard Architecture

This describes the end-to-end flow ReleaseGuard is being built toward
(per spec Part 2), and which pieces exist today (Phases 0-3) versus which
are planned for later phases. It will be expanded as each phase lands
rather than rewritten from scratch.

## Target flow (full system, spec Part 2)

```
Git commit
  -> CI: test / build / scan
  -> immutable image (tag + digest)
  -> deployment request
  -> stable + canary workloads (Kubernetes)
  -> telemetry collector/query layer (OTel -> Prometheus)
  -> evaluator
       policy engine
       | statistics / window aggregator
       | decision state machine
  -> Kubernetes deploy/rollback adapter
  -> audit/event log
  -> dashboard / CLI
```

## What exists today (Phases 0-3)

```
Git commit
  -> CI: test / build / push          [demo-service/, .github/workflows/ci.yml]
  -> immutable image (tag + digest)   [Dockerfile; verified locally, see
                                        experiments/raw/phase0-image-build-evidence.md]
  -> deployment request                [deploy/kubernetes/, deploy/local-cluster/]
  -> stable + canary workloads         [deploy/kubernetes/base/deployment.yaml,
     (Kubernetes)                       deployment-canary.yaml; see
                                        "Deployment topology" below]
  -> telemetry collector/query layer   [observability/otel/ (traces),
     (OTel -> Prometheus)               observability/prometheus/ (metrics
                                        scrape + PromQL); see ADR 0009]
```

Everything from "evaluator" onward (policy engine, decision state
machine, Kubernetes deploy/rollback adapter, audit log, dashboard/CLI)
does not exist yet -- that is `controller/`, starting Phase 4. The
demo-service already exposes Prometheus-format metrics on `/metrics`
(route/method/status/version/track-labeled request counters and a latency
histogram, plus a dependency-failure counter) and OTel trace spans (see
ADR 0009), all labeled by `track`/`version`, so Phase 4's evaluator has
real per-identity telemetry to query rather than a fixture to design
around.

## Why the demo-service looks the way it does

The demo-service (`demo-service/`) is deliberately not "the project" --
see spec §1.1: *"the important implementation is the decision/controller
logic and evidence pipeline -- not a collection of YAML files."* Its only
job is to be a believable canary target:

- `/health`, `/version`, `/work` cover the spec's required endpoints
  (§1.2), plus `/metrics` for Prometheus scraping.
- `/work` calls a mock dependency (`internal/dependency`) whose failure
  probability and "hard down" state are both configurable, so the same
  binary and image can play every role the spec's failure-injection
  scenarios need (§1.8: healthy, latency regression, error regression,
  missing telemetry, brief blip) purely via environment variables on the
  deployment -- no code changes, no separate images per scenario.
- `/health` never fails just because a fault is injected (see the comment
  on `healthResponse` in `internal/server/handlers.go`). This is a
  deliberate separation of concerns: kubelet's liveness/readiness checks
  answer "is the process alive," while ReleaseGuard's evaluator (Phase 4+)
  answers "is this release healthy" from telemetry. Conflating the two
  would mean Kubernetes silently restarts or removes the very pods a
  failure-injection scenario is trying to keep unhealthy-but-running so
  the controller can observe and react to them.

## Deployment topology (Phases 1-3)

A single Kubernetes namespace (`releaseguard`) contains two Deployments --
`demo-service-stable` (2 replicas, `RELEASE_VERSION=v1.0.0`) and
`demo-service-canary` (1 replica, `RELEASE_VERSION=v1.1.0`) -- plus three
ClusterIP Services:

- `demo-service-stable` / `demo-service-canary` each select only their own
  track, for direct single-track access (smoke-testing, debugging).
- `demo-service` selects both tracks via the shared
  `app.kubernetes.io/name: demo-service` label with no track filter --
  this is ADR 0003's traffic-split mechanism in effect: kube-proxy
  load-balances across every matching Endpoint roughly uniformly, so the
  2:1 stable:canary replica ratio yields an approximate 66/33 split with
  no service mesh, ingress controller, or extra CRDs.
  `tests/e2e/canary_smoke_test.sh` verifies this Service's Endpoints
  actually contain pod IPs from both tracks (not just that the YAML
  declares it).

See [ADR 0003](decisions/0003-traffic-splitting-mechanism.md) for the full
design rationale.

Phase 3 is "manual canary": nothing here decides promote/rollback yet
(that starts Phase 4's evaluator and lands as automated action in
Phase 6). The one manual lever is
`deploy/local-cluster/canary-scenario.sh healthy|unhealthy`, which patches
the canary Deployment's fault-injection env vars (`ERROR_RATE`,
`EXTRA_LATENCY_MS`, `DEPENDENCY_DOWN` -- the same knobs
`demo-service/internal/config/config.go` has validated since Phase 0) and
waits for the resulting rollout, so a human can generate either healthy or
unhealthy canary telemetry on demand -- satisfying Phase 3's exit
criterion directly. The full scripted failure-injection scenario matrix
(latency-only, error-only, saturation, missing telemetry) is Phase 7's
FR5 deliverable, not this one.

Manifests are plain Kubernetes YAML managed with Kustomize (not Helm) --
see [ADR 0002](decisions/0002-manifest-tooling-kustomize.md). The `local`
overlay (`deploy/kubernetes/overlays/local`) remains the only overlay;
Phase 3 added the canary Deployment and the two new Services as base
resources rather than a separate overlay, since both tracks always deploy
together locally -- there is no scenario yet where you would want the
base without the canary.

## Controller (Phases 4-5)

`controller/` is a separate Go module (`releaseguard/controller`,
`controller/go.mod`) from `demo-service/` -- a standalone service, not a
CRD/operator (per [ADR 0006](decisions/0006-standalone-vs-operator.md)).
Phase 4 built the policy/evaluator core; Phase 5 built the release state
machine and audit log on top of it. Phase 6 (not yet built) adds the
Kubernetes deploy/rollback adapter that will make this real against a live
cluster; everything described below already runs today against recorded
telemetry fixtures, with zero cluster or network dependency.

```
controller/
  cmd/controller/        CLI: `validate` (policy) and `simulate` (scenario -> full rollout)
  internal/policy/       typed policy schema + validation (FR4)
  internal/metrics/      Querier interface: Prometheus (real) + Fixture (tests/CLI) -- ADR 0004
  internal/evaluator/    single-window evaluation: policy + Querier -> WindowResult
  internal/rollout/      the release lifecycle state machine + Action interface
  internal/audit/        JSONL audit log (ADR 0005) + in-memory logger for tests
  internal/scenario/     loads a recorded-telemetry fixture, drives evaluator+rollout end to end
  policy/examples/       the default demo-service policy (values flagged for human review)
  scenarios/             spec §1.8's five demo scenarios (A-E) as fixture files
```

### Policy (`internal/policy`) -- Phase 4

A typed YAML/JSON policy (single schema for both formats via
`sigs.k8s.io/yaml`, unknown fields rejected) defines: evaluation window +
cadence, minimum sample count/traffic, consecutive healthy/unhealthy
window counts, maximum rollout duration, the action on missing telemetry
(`pause` or `rollback` -- no `promote` value exists in the schema), and a
list of signals, each an absolute threshold (required) plus an optional
canary-vs-baseline relative guardrail
([ADR 0007](decisions/0007-absolute-vs-relative-thresholds.md): additive,
never a replacement). A signal's `query` is a `text/template` string
(`{{.Track}}`, `{{.Window}}`) rendered once per track evaluated -- see
[`docs/slo-policy.md`](slo-policy.md) for the full schema and which
values in `controller/policy/examples/demo-service-default.yaml` are still
flagged for human review.

### Metrics query (`internal/metrics`) -- Phase 4, ADR 0004

`Querier` is the interface the evaluator depends on: `InstantQuery` and
`RangeQuery`, matching ADR 0004's two PromQL shapes.`Prometheus` wraps
`github.com/prometheus/client_golang`'s v1 API client against a live
server (tested against an `httptest.Server` speaking the real Prometheus
HTTP API response shape -- `controller/internal/metrics/prometheus_test.go`
-- since `tests/integration/` cannot import an `internal/` package outside
its own module's import-path prefix; see `tests/README.md`). `Fixture` is
the in-memory, map-backed implementation tests and the CLI's `simulate`
command use instead. Both surface `ErrNoData` for "no series" and for
Prometheus's own NaN representation of a 0/0 division -- never a value the
evaluator could mistake for real data.

### Evaluator (`internal/evaluator`) -- Phase 4

`Evaluator.Evaluate(ctx, track, now)` runs the sample-count gate (if
configured) then every signal, and classifies the window `HEALTHY`,
`DEGRADED`, or `INCONCLUSIVE`. A missing *required* signal, a missing
sample count, or a missing relative-guardrail baseline all classify the
window `INCONCLUSIVE` -- never `HEALTHY` -- regardless of every other
signal's value (Claude Code handoff rule 1, enforced here structurally,
not by convention). A signal marked `optional: true` is the one exception:
its own absence doesn't force `INCONCLUSIVE`.

### Rollout state machine (`internal/rollout`) -- Phase 5

`Rollout` implements the spec's full lifecycle chain exactly:

```
PENDING -> DEPLOYING -> OBSERVING -> HEALTHY/DEGRADED/INCONCLUSIVE ->
PROMOTING/ROLLING_BACK/PAUSED -> COMPLETED/FAILED
```

as an explicit, exhaustive `allowedTransitions` table (`state.go`) --
every legal edge is listed, so an illegal transition is rejected outright
rather than merely unlikely. The table's central safety property (rule 1
again): the *only* edge leading to `PROMOTING` originates at `HEALTHY`;
`INCONCLUSIVE` and `DEGRADED` have no direct path to it, which
`TestAllowedTransitions_OnlyHealthyLeadsToPromoting` checks over every
table entry, not a spot check. `Rollout` is clock-free -- every method
that can transition state takes an explicit `now time.Time` rather than
reading the wall clock, so the exact same code drives an instant
simulation and a real-time Phase 6 reconcile loop. `RecordWindow` tracks
independent consecutive-window streaks (healthy/degraded/inconclusive,
each reset by any other classification) against the policy's configured
thresholds, and is only callable while the rollout is in an observing
state -- calling it again after a terminal state is reached is rejected,
which is what keeps a duplicate/late reconcile from invoking
`Action.Promote`/`Rollback` a second time (rule 2, idempotency). See
[ADR 0010](decisions/0010-missing-telemetry-and-pause-semantics.md) for
`on_missing_telemetry`'s default and what `PAUSED -> COMPLETED` means
without a human-in-the-loop resume API yet.

`Action` (Promote/Rollback) is the one seam Phase 6's real Kubernetes
adapter will fill in -- Phase 5 tests it against `NoopAction`,
`RecordingAction` (call-counting, for idempotency assertions) and
`FailingAction` (the `Promoting/RollingBack -> Failed` path).

### Audit log (`internal/audit`) -- Phase 5, ADR 0005

Every state transition is one `audit.Event`. `JSONLLogger` appends one
JSON object per line with an `fsync` after each write (crash-safe by
construction: a torn write can at worst leave one incomplete trailing
line, and `ReadJSONL` tolerates exactly that). `InMemoryLogger` backs
tests. Evidence (the full per-signal `WindowResult`) travels with every
classification event, not just a bare state name -- this is what actually
answers FR4's "why did ReleaseGuard make this decision?"

### Scenario runner (`internal/scenario`) + CLI -- ties it together

`scenario.Run` loads a recorded-telemetry fixture (`Tick`s: per-signal
values for canary and, where needed, stable) and drives a real `Rollout`
through it window by window, using a real `Evaluator` against a
per-tick `Fixture` -- the exact same call sequence a live Phase 6 loop
will make, just with simulated instead of real time and data.
`controller/scenarios/` holds one fixture per spec §1.8 demo scenario
(A-E); `controller/internal/scenario/scenario_test.go` runs all five
against the real default policy and asserts the exact expected decision,
twice each, to prove determinism. `cmd/controller simulate` is the same
code path exposed as the "local reproducibility command" Part 4 of the
spec requires -- see
[`experiments/raw/phase4-5-scenario-evidence.md`](../experiments/raw/phase4-5-scenario-evidence.md)
for a real, captured run of all five.

## Kubernetes adapter (Phase 6, not yet built)

The `Action` interface `internal/rollout` already depends on is expected
to be a thin, idempotent wrapper around patching the stable and canary
Deployments' replica counts, consistent with the replica-count
traffic-split design in ADR 0003 -- nothing in Phase 4/5's design should
need to change to plug it in.
