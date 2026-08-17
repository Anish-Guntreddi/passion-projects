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

## Controller (not yet built)

`controller/` does not exist yet. When it lands starting Phase 4, it will
be a standalone Go service (not a CRD/operator -- see
[ADR 0006](decisions/0006-standalone-vs-operator.md)) that:

1. Loads a typed policy (YAML/JSON, absolute thresholds first --
   [ADR 0007](decisions/0007-absolute-vs-relative-thresholds.md)).
2. Queries Prometheus via instant + range HTTP queries behind a
   `MetricsQuerier` interface
   ([ADR 0004](decisions/0004-prometheus-query-strategy.md)), which is
   what makes its decision logic unit-testable against fixture data
   without a live cluster.
3. Runs a decision state machine (PENDING -> DEPLOYING -> OBSERVING ->
   HEALTHY/DEGRADED/INCONCLUSIVE -> PROMOTING/ROLLING_BACK/PAUSED ->
   COMPLETED/FAILED, per spec §1.3 FR4) and writes every transition to an
   append-only JSONL audit log
   ([ADR 0005](decisions/0005-audit-persistence.md)).
4. Applies promote/rollback decisions via an idempotent Kubernetes adapter
   (Phase 6) -- expected to be a thin wrapper around patching the stable
   and canary Deployments' replica counts, consistent with the
   replica-count traffic-split design in ADR 0003.

This section will be rewritten with real detail once that code exists;
it is included here now so the Phase 0-1 architecture is legible in the
context of where it's headed, not just what it is today.
