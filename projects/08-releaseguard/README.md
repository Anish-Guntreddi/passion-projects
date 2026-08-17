# ReleaseGuard

SLO-aware progressive-delivery and automated rollback controller. See
[`../../08-releaseguard-spec.md`](../../08-releaseguard-spec.md) for the
full PRD, tech-stack plan, and roadmap this project implements.

**Status:** Phases 0-5 of the roadmap are implemented: demo service + CI,
local Kubernetes deployment foundation, OTel/Prometheus telemetry, manual
stable+canary split (Phases 0-3), and now the controller itself --
typed policy schema, metrics-query interface, evaluator, and the full
release lifecycle state machine with an append-only audit log (Phases
4-5) -- see §1.1 of the spec: *"the important implementation is the
decision/controller logic and evidence pipeline -- not a collection of
YAML files."* The controller (`controller/`) runs entirely against
recorded telemetry fixtures today, with zero cluster or live-Prometheus
dependency; Phase 6 adds the Kubernetes deploy/rollback adapter that
makes its decisions real.

## What's here today

- `demo-service/` -- a small instrumented Go HTTP service
  (`/health`, `/version`, `/work`, `/metrics`) with env-var-controlled
  fault injection (latency, error rate, dependency outage, memory
  pressure), OTel trace spans, and Prometheus metrics -- ReleaseGuard's
  canary target.
- `deploy/kubernetes/` -- Kustomize base + local overlay deploying both
  the `stable` (2 replicas) and `canary` (1 replica) tracks of
  demo-service, plus the shared Service that implements ADR 0003's
  replica-count traffic split.
- `deploy/local-cluster/` -- kind cluster config, lifecycle scripts, and
  `canary-scenario.sh` (flip the canary track healthy/unhealthy).
- `observability/` -- OTel Collector (traces) and Prometheus (metrics
  scrape + PromQL) Kustomize manifests -- see ADR 0009.
- `tests/e2e/smoke_test.sh` / `canary_smoke_test.sh` -- verify a fresh
  local deployment actually serves traffic correctly on both tracks, and
  that the shared Service's routing genuinely spans both.
- `controller/` -- the ReleaseGuard controller itself (a separate Go
  module, `releaseguard/controller`): typed policy schema + validation,
  a Prometheus/fixture-backed metrics-query interface, the single-window
  evaluator, the full release lifecycle state machine, and a JSONL audit
  log -- see [What's in `controller/`](#whats-in-controller-phases-4-5)
  below and [`docs/architecture.md`](docs/architecture.md#controller-phases-4-5).
- `.github/workflows/ci.yml` -- test + build + Trivy vulnerability scan +
  push pipeline (see [CI note](#ci-workflow-location) below).
- `docs/decisions/` -- ADRs for the spec's open decisions (D1-D7), Phase
  2's OTel/Prometheus scope clarification (0009), and Phase 4-5's
  missing-telemetry/pause design (0010).

## Prerequisites

- Docker (or another OCI-compatible engine the `docker` CLI talks to)
- Go 1.22+
- [kind](https://kind.sigs.k8s.io/) v0.20+
- `kubectl` v1.29+ (bundles a compatible Kustomize; no separate Kustomize
  install needed -- see [ADR 0002](docs/decisions/0002-manifest-tooling-kustomize.md))
- A POSIX `bash` -- **on Windows this means WSL2 or Git Bash, not
  PowerShell/cmd.exe.** Every script this project ships
  (`deploy/local-cluster/*.sh`, `tests/e2e/smoke_test.sh`, and the
  `Makefile` targets that shell out to them) is a bash script using
  bash-only features (`set -euo pipefail`, `${BASH_SOURCE[0]}`, `trap`,
  process substitution). `make cluster-up` on a fresh Windows clone will
  fail outright without one of these installed; this repo was developed
  and is verified against WSL2 Ubuntu.

All of the above are free, locally-installable tools; no cloud account or
paid service is required for anything in this repo (see
[ADR 0008](docs/decisions/0008-terraform-cloud-deferred.md)).

## Quickstart

```bash
# Run the service locally, no container/cluster needed:
make run
# in another shell:
curl localhost:8080/health
curl localhost:8080/version
curl localhost:8080/work
curl localhost:8080/metrics

# Run the test suite:
make test

# Build the container image:
make docker-build

# Stand up a local kind cluster, deploy stable + canary, and smoke-test both:
make cluster-up

# Flip the canary to an unhealthy config (elevated errors + latency),
# generate some traffic against it, then flip it back:
make canary-unhealthy
make canary-healthy

# Tear it down:
make cluster-down

# Controller (Phases 4-5) -- no cluster needed, runs against fixtures:
make controller-test              # go vet + go test -race -cover for controller/
make controller-validate-policy   # validate the default policy file
make controller-simulate          # run scenario A (healthy canary) through the state machine end to end
```

`make cluster-up` is the single command Phase 1's exit criterion ("fresh
clone can deploy locally") is checked against: it creates (or reuses) the
`releaseguard-local` kind cluster, builds the demo-service image, loads it
directly into the cluster (no registry needed locally), applies
`deploy/kubernetes/overlays/local` (stable + canary + OTel Collector +
Prometheus), waits for both Deployments' rollouts, and runs both
`tests/e2e/smoke_test.sh` and `tests/e2e/canary_smoke_test.sh`.

`make canary-healthy` / `make canary-unhealthy` wrap
`deploy/local-cluster/canary-scenario.sh`, Phase 3's manual lever for
generating healthy vs. unhealthy canary telemetry on demand -- see
[docs/architecture.md](docs/architecture.md#deployment-topology-phases-1-3).

## Demo-service configuration

The demo-service reads its behavior entirely from environment variables
(see `demo-service/internal/config/config.go` for full validation rules):

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | HTTP listen port |
| `RELEASE_TRACK` | `stable` | `stable` or `canary` -- attached to every metric/response |
| `RELEASE_VERSION` | `dev` | Deployment-level version label (see `/version`'s `release_version` vs `image_version` distinction) |
| `BASE_LATENCY_MS` | `20` | Baseline simulated `/work` latency |
| `LATENCY_JITTER_MS` | `10` | Random jitter added on top of the baseline |
| `EXTRA_LATENCY_MS` | `0` | Constant extra latency -- used to simulate a latency regression |
| `ERROR_RATE` | `0.0` | Probability in `[0,1]` that `/work`'s simulated dependency call fails |
| `DEPENDENCY_DOWN` | `false` | Forces every `/work` call to fail -- simulates a hard dependency outage |
| `MEMORY_PRESSURE_MB` | `0` | Memory allocated and retained at startup, to simulate resource pressure |

These knobs exist so Phase 7's failure-injection scenarios can reproduce a
specific canary failure mode by changing environment variables on a
deployment, without touching code.

`/health` always returns HTTP 200 as long as the process is alive, even
when a fault is injected -- see the comment on `healthResponse` in
`demo-service/internal/server/handlers.go` for why (short version:
conflating "process alive" with "release healthy" would make Kubernetes
itself interfere with the failure-injection scenarios this project exists
to run).

## What's in `controller/` (Phases 4-5)

`controller/` is its own Go module (`releaseguard/controller`,
`controller/go.mod` -- deliberately separate from `demo-service/`'s
module) implementing the actual decision/evidence pipeline §1.1 of the
spec calls the important part of this project. Everything below runs
against recorded telemetry fixtures with no cluster or live Prometheus
required; see [`docs/architecture.md`](docs/architecture.md#controller-phases-4-5)
for the full design and [`docs/slo-policy.md`](docs/slo-policy.md) for the
policy schema.

```bash
cd controller
go run ./cmd/controller validate -policy policy/examples/demo-service-default.yaml
go run ./cmd/controller simulate \
    -policy policy/examples/demo-service-default.yaml \
    -scenario scenarios/a-healthy-promotion.json
```

- **`internal/policy`** -- typed policy schema (YAML or JSON, one schema
  for both via `sigs.k8s.io/yaml`), fully validated at load time (unknown
  fields rejected, every threshold/window/count checked, `on_missing_
  telemetry` restricted to `pause`/`rollback` -- there is no `promote`
  value in the schema at all).
- **`internal/metrics`** -- the `Querier` interface (ADR 0004: instant +
  range PromQL behind one interface) with a real `Prometheus`
  implementation (`github.com/prometheus/client_golang`, tested against a
  fake HTTP server speaking the real API response shape) and an in-memory
  `Fixture` implementation tests and the CLI use.
- **`internal/evaluator`** -- single-window evaluation: runs the
  sample-count gate then every signal, classifies the window
  `HEALTHY`/`DEGRADED`/`INCONCLUSIVE`. A missing required signal, sample
  count, or relative-guardrail baseline always classifies `INCONCLUSIVE`,
  never `HEALTHY` (Claude Code handoff rule 1).
- **`internal/rollout`** -- the full lifecycle state machine (`PENDING ->
  DEPLOYING -> OBSERVING -> HEALTHY/DEGRADED/INCONCLUSIVE ->
  PROMOTING/ROLLING_BACK/PAUSED -> COMPLETED/FAILED`) as an explicit,
  exhaustive transition table -- only `HEALTHY` has an edge to
  `PROMOTING`, checked over every table entry by
  `TestAllowedTransitions_OnlyHealthyLeadsToPromoting`. Clock-free (every
  transition takes an explicit `now`), so the same code drives instant
  simulations and a future real-time reconcile loop.
- **`internal/audit`** -- append-only JSONL audit log (ADR 0005), one
  event per transition with the full evidence attached, `fsync`'d on
  every write.
- **`internal/scenario`** + **`scenarios/*.json`** -- the five spec §1.8
  demo scenarios (A healthy->promote, B latency regression->rollback, C
  error regression->rollback, D missing telemetry->pause (never
  promote), E transient blip->no overreaction) as recorded-telemetry
  fixtures, run end to end through the real evaluator and state machine.
  See [`experiments/raw/phase4-5-scenario-evidence.md`](experiments/raw/phase4-5-scenario-evidence.md)
  for a real, captured run of all five against the actual default policy
  file.
- **`policy/examples/demo-service-default.yaml`** -- the default canary
  policy for demo-service. Every threshold value is explicitly flagged
  inline as needing human review (spec §1.9: "policy threshold values are
  human-review decisions") -- see `docs/slo-policy.md`.

`docs/decisions/0010-missing-telemetry-and-pause-semantics.md` documents
the two non-obvious design calls Phase 4-5 had to make: `on_missing_
telemetry`'s safe default, and what `PAUSED -> COMPLETED` means with no
human-in-the-loop resume API yet.

## Repository layout

Adapted from the spec's suggested structure (Part 2); `controller/`
(Phases 4-5) is now populated -- see above for what's in it. Directories
still not populated (e.g. a future `internal/k8s/` Kubernetes adapter,
Phase 6) are omitted here rather than scaffolded empty.

```
demo-service/
  cmd/demo-service/        entrypoint
  internal/config/         env-var config loading + validation
  internal/dependency/     mock downstream dependency (fault injection)
  internal/server/         HTTP handlers, Prometheus instrumentation
  internal/telemetry/      OTel tracer provider + OTLP export
  internal/version/        build-time version metadata (ldflags)
  Dockerfile
deploy/
  kubernetes/base/         K8s manifests: Namespace, stable + canary
                            Deployments, per-track + shared Services
  kubernetes/overlays/local/  kind-specific overlay (base + observability/)
  local-cluster/           kind config, setup/teardown, canary-scenario.sh
observability/
  otel/                    OTel Collector Kustomize manifests
  prometheus/              Prometheus server Kustomize manifests
controller/
  cmd/controller/          CLI entrypoint: validate, simulate
  internal/policy/         typed policy schema + validation
  internal/metrics/        Querier interface: Prometheus (real) + Fixture
  internal/evaluator/      single-window evaluation
  internal/rollout/        release lifecycle state machine + Action interface
  internal/audit/          JSONL audit log + in-memory logger
  internal/scenario/       recorded-telemetry fixture -> full rollout runner
  policy/examples/         default demo-service policy (flagged for review)
  scenarios/               spec §1.8's five demo scenarios as fixtures
tests/
  e2e/smoke_test.sh          Phase 1 stable-track deployment smoke test
  e2e/canary_smoke_test.sh   Phase 3 canary + split-routing smoke test
  integration/, failure/, unit/   reserved (see tests/README.md for why
                                   Phase 4's Prometheus-adapter test is
                                   co-located in controller/ instead)
docs/decisions/            ADRs (D1-D7 + manifest tooling + OTel/Prometheus
                            scope + Phase 4-5 missing-telemetry/pause design)
docs/slo-policy.md         Phase 4 policy schema reference
experiments/raw/           committed, measured evidence (build/digest logs, scenario runs)
.github/workflows/ci.yml   test + build + push pipeline
```

## CI workflow location

`.github/workflows/ci.yml` lives inside this project directory
(`projects/08-releaseguard/.github/workflows/ci.yml`) because this project
currently lives inside the `passion-projects` monorepo, but GitHub Actions
only discovers workflows under the true repository root. Until this
project is split into its own repository (per the portfolio's stated
intent), the workflow needs to be copied or referenced from the monorepo's
root `.github/workflows/` for it to actually execute on GitHub -- the
`paths:` filters in the workflow already scope it to this project, so that
copy is a no-op in behavior. See the workflow file's header comment for
details, and `experiments/raw/phase0-image-build-evidence.md` for a real,
measured run of the same build-tag-push-digest sequence executed locally
against a throwaway registry, and `experiments/raw/phase0-trivy-scan-evidence.md`
for the same treatment of the workflow's Trivy vulnerability-scan gate
(FR1) -- both since this environment has no git-push access to actually
trigger the GitHub-hosted job.

**Known gap:** `.github/workflows/ci-releaseguard.yml` (at the monorepo
root, outside this project directory -- see above) only runs
`demo-service`'s test/build/scan job today; it does not yet run `go vet`/
`go test` for `controller/`. Adding a `controller` test job (mirroring the
existing `test` job's shape, pointed at `controller/go.mod`) is a small,
mechanical follow-up outside this phase's assigned directory
(`projects/08-releaseguard/`) -- flagged here rather than made silently,
per this repo's project-directory boundary. `make test` (and CI's own
future `controller` job once added) is the authoritative local check in
the meantime.

## Testing

```bash
make test               # go vet + go test -race -cover, BOTH demo-service and controller modules
make controller-test    # go vet + go test -race -cover, controller module only
make smoke-test         # requires a running cluster with the stable track deployed
make smoke-test-canary  # requires a running cluster with the canary track deployed
```

`tests/e2e/smoke_test.sh` checks `/health`, `/version`, `/work` and
`/metrics` both through the Service (routing sanity) and against every
individual stable pod directly (a Service port-forward only ever reaches
one backing pod, so per-pod checks are what actually catch a replica that
is individually broken). `tests/e2e/canary_smoke_test.sh` does the same
for the canary track, plus verifies the shared `demo-service` Service's
Endpoints actually include pod IPs from both tracks -- proof ADR 0003's
traffic split is real routing, not just declared YAML.

Current demo-service coverage (measured, not aspirational):
`internal/config` 98.3%, `internal/dependency` 100.0%, `internal/server`
92.1%, `internal/telemetry` 83.3%. The full, unedited
`go test -race -cover -v` run backing these
numbers is committed at
[`experiments/raw/phase0-test-coverage.md`](experiments/raw/phase0-test-coverage.md)
-- re-run `make test` and regenerate that file (rather than hand-editing
either) whenever the test suite changes, so the two never drift apart.
CI's image vulnerability scan (Trivy, gating on CRITICAL/HIGH fixable
findings) has a similar measured artifact at
[`experiments/raw/phase0-trivy-scan-evidence.md`](experiments/raw/phase0-trivy-scan-evidence.md).

Controller (Phase 4-5) coverage (measured, `go test ./... -race -cover`
from `controller/`): `internal/policy` 92.1%, `internal/evaluator` 92.7%,
`internal/metrics` 86.2%, `internal/audit` 85.7%, `internal/rollout`
83.9%, `internal/scenario` 81.0% -- 80 top-level tests, all passing,
race-clean.
All five spec §1.8 demo scenarios (A-E) run end to end against the real
default policy file and produce the exact decision each scenario is
meant to demonstrate, twice each (determinism check) -- see
[`experiments/raw/phase4-5-scenario-evidence.md`](experiments/raw/phase4-5-scenario-evidence.md)
for the real, captured `controller simulate` output backing this.

## Decisions

See [`docs/decisions/`](docs/decisions/) for ADRs covering every open
decision in spec §1.9 (D1-D7), the Helm-vs-Kustomize manifest-tooling
choice, and Phase 4-5's missing-telemetry/pause design (0010). Every ADR
for a spec-flagged decision (D1-D7) adopts the spec's recommended default.

## Roadmap status

| Phase | Status |
|---|---|
| 0 -- Demo service + CI | Done |
| 1 -- Local deployment foundation | Done |
| 2 -- Telemetry (OTel traces, Prometheus scrape/query) | Done |
| 3 -- Manual canary (stable+canary split, healthy/unhealthy toggle) | Done |
| 4 -- Policy/evaluator core | Done |
| 5 -- Release state machine | Done |
| 6+ | Not started |

See [`../../08-releaseguard-spec.md`](../../08-releaseguard-spec.md) Part 3
for the full roadmap.
