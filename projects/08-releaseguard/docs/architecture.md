# ReleaseGuard Architecture

This describes the end-to-end flow ReleaseGuard is being built toward
(per spec Part 2), and which pieces exist today (Phases 0-1) versus which
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

## What exists today (Phases 0-1)

```
Git commit
  -> CI: test / build / push          [demo-service/, .github/workflows/ci.yml]
  -> immutable image (tag + digest)   [Dockerfile; verified locally, see
                                        experiments/raw/phase0-image-build-evidence.md]
  -> deployment request                [deploy/kubernetes/, deploy/local-cluster/]
  -> stable workload (Kubernetes)      [deploy/kubernetes/base/deployment.yaml,
                                         one track only -- no canary yet]
```

Everything from "telemetry collector/query layer" onward does not exist
yet. The demo-service already exposes Prometheus-format metrics on
`/metrics` (route/method/status/version/track-labeled request counters
and a latency histogram, plus a dependency-failure counter) so that Phase
2 only has to stand up a Prometheus server and point it at the running
pods -- the instrumentation itself does not need retrofitting.

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

## Deployment topology (Phase 1)

A single Kubernetes namespace (`releaseguard`) contains one Deployment
(`demo-service-stable`, 2 replicas) and one ClusterIP Service selecting
it. There is no canary workload yet (Phase 3) and no traffic-splitting
mechanism wired up yet, though its design is already decided --
see [ADR 0003](decisions/0003-traffic-splitting-mechanism.md)
(replica-count-based split behind a shared Service, once a canary
Deployment exists alongside the stable one).

Manifests are plain Kubernetes YAML managed with Kustomize (not Helm) --
see [ADR 0002](decisions/0002-manifest-tooling-kustomize.md). The `local`
overlay is the only overlay that exists today; a `canary` overlay (or
Deployment variant) is a Phase 3 addition.

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
