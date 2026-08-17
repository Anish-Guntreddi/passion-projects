# ReleaseGuard

SLO-aware progressive-delivery and automated rollback controller. See
[`../../08-releaseguard-spec.md`](../../08-releaseguard-spec.md) for the
full PRD, tech-stack plan, and roadmap this project implements.

**Status:** Phases 0-1 of the roadmap are implemented (demo service + CI,
local Kubernetes deployment foundation). The controller itself
(policy/evaluator/state-machine/audit) lands starting Phase 4 -- see
§1.1 of the spec: *"the important implementation is the decision/
controller logic and evidence pipeline -- not a collection of YAML
files."* Phases 0-1 exist to give the controller something real to
observe once it's built.

## What's here today

- `demo-service/` -- a small instrumented Go HTTP service
  (`/health`, `/version`, `/work`, `/metrics`) with env-var-controlled
  fault injection (latency, error rate, dependency outage, memory
  pressure), used later as ReleaseGuard's canary target.
- `deploy/kubernetes/` -- Kustomize base + local overlay deploying the
  `stable` track of demo-service.
- `deploy/local-cluster/` -- kind cluster config and lifecycle scripts.
- `tests/e2e/smoke_test.sh` -- verifies a fresh local deployment actually
  serves traffic correctly.
- `.github/workflows/ci.yml` -- test + build + Trivy vulnerability scan +
  push pipeline (see [CI note](#ci-workflow-location) below).
- `docs/decisions/` -- ADRs for the spec's open decisions (D1-D7).

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

# Stand up a local kind cluster, deploy the stable track, and smoke-test it:
make cluster-up

# Tear it down:
make cluster-down
```

`make cluster-up` is the single command Phase 1's exit criterion ("fresh
clone can deploy locally") is checked against: it creates (or reuses) the
`releaseguard-local` kind cluster, builds the demo-service image, loads it
directly into the cluster (no registry needed locally), applies
`deploy/kubernetes/overlays/local`, waits for the rollout, and runs
`tests/e2e/smoke_test.sh`.

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

## Repository layout

Adapted from the spec's suggested structure (Part 2); directories not yet
populated (e.g. `controller/`, `observability/prometheus/`) are omitted
here rather than scaffolded empty, and will appear starting the phase that
needs them.

```
demo-service/
  cmd/demo-service/        entrypoint
  internal/config/         env-var config loading + validation
  internal/dependency/     mock downstream dependency (fault injection)
  internal/server/         HTTP handlers, Prometheus instrumentation
  internal/version/        build-time version metadata (ldflags)
  Dockerfile
deploy/
  kubernetes/base/         plain K8s manifests (Namespace, Deployment, Service)
  kubernetes/overlays/local/  kind-specific Kustomize overlay
  local-cluster/           kind config + setup/teardown scripts
tests/
  e2e/smoke_test.sh        Phase 1 deployment smoke test
  integration/, failure/, unit/   reserved for Phase 2+ (see tests/README.md)
docs/decisions/            ADRs (D1-D7 + manifest tooling)
experiments/raw/           committed, measured evidence (build/digest logs, later benchmark data)
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

## Testing

```bash
make test          # go vet + go test -race -cover, demo-service module
make smoke-test     # requires a running cluster with the stable track deployed
```

`tests/e2e/smoke_test.sh` checks `/health`, `/version`, `/work` and
`/metrics` both through the Service (routing sanity) and against every
individual stable pod directly (a Service port-forward only ever reaches
one backing pod, so per-pod checks are what actually catch a replica that
is individually broken).

Current demo-service coverage (measured, not aspirational):
`internal/config` 98.0%, `internal/dependency` 100.0%, `internal/server`
90.0%. The full, unedited `go test -race -cover -v` run backing these
numbers is committed at
[`experiments/raw/phase0-test-coverage.md`](experiments/raw/phase0-test-coverage.md)
-- re-run `make test` and regenerate that file (rather than hand-editing
either) whenever the test suite changes, so the two never drift apart.
CI's image vulnerability scan (Trivy, gating on CRITICAL/HIGH fixable
findings) has a similar measured artifact at
[`experiments/raw/phase0-trivy-scan-evidence.md`](experiments/raw/phase0-trivy-scan-evidence.md).

## Decisions

See [`docs/decisions/`](docs/decisions/) for ADRs covering every open
decision in spec §1.9 (D1-D7) plus the Helm-vs-Kustomize manifest-tooling
choice. Every ADR adopts the spec's recommended default.

## Roadmap status

| Phase | Status |
|---|---|
| 0 -- Demo service + CI | Done |
| 1 -- Local deployment foundation | Done |
| 2 -- Telemetry (OTel/Prometheus server, dashboards) | Not started |
| 3+ | Not started |

See [`../../08-releaseguard-spec.md`](../../08-releaseguard-spec.md) Part 3
for the full roadmap.
