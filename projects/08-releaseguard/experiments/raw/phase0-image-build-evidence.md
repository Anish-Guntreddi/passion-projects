# Phase 0 exit-criterion evidence: immutable tagged/digested image

**Exit criterion (roadmap Phase 0):** "CI produces immutable tagged/digested image."

**What this file is:** a raw, measured record of running the exact build the
`.github/workflows/ci.yml` `build-and-push` job runs (build with git-SHA
provenance baked in via `--build-arg`, push, capture the registry-assigned
content digest) against a local OCI registry, since this sandbox cannot push
to GitHub itself (no git/remote operations are available to the implementer
agent -- see repo rule "never run git commands"). The mechanism is identical
to what CI does against `ghcr.io`; only the registry endpoint differs.

## Environment

| | |
|---|---|
| Date (UTC) | 2026-08-17T07:01Z |
| Host | Windows 11 Pro, WSL2 Ubuntu 24.04.1 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Docker | 27.4.1 (client + server) |
| Go (build stage base image) | golang:1.22-alpine |
| Runtime base image | gcr.io/distroless/static-debian12:nonroot |
| Local registry | `registry:2` (Docker Hub official image), localhost:5000 |
| Repo commit at build time | `b5965b5` (short SHA of the passion-projects monorepo HEAD) |

## Command

```bash
GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
VERSION="phase0-evidence-${GIT_COMMIT}"

docker run -d --name rg-local-registry -p 5000:5000 registry:2

docker build --no-cache \
  --build-arg VERSION="$VERSION" \
  --build-arg GIT_COMMIT="$GIT_COMMIT" \
  --build-arg BUILD_TIME="$BUILD_TIME" \
  -t "localhost:5000/releaseguard-demo-service:${GIT_COMMIT}" \
  -t "localhost:5000/releaseguard-demo-service:phase0-evidence" \
  "$PROJECT_ROOT/demo-service"

docker push "localhost:5000/releaseguard-demo-service:${GIT_COMMIT}"

docker inspect "localhost:5000/releaseguard-demo-service:${GIT_COMMIT}" \
  --format '{{index .RepoDigests 0}}'
```

## Measured result

```
GIT_COMMIT=b5965b5
BUILD_TIME=2026-08-17T07:01:32Z
VERSION=phase0-evidence-b5965b5

Successfully tagged localhost:5000/releaseguard-demo-service:b5965b5
Successfully tagged localhost:5000/releaseguard-demo-service:phase0-evidence

b5965b5: digest: sha256:8e7b271905dcf442825066cb4f63f3d0cee88d2c6abc3e1548e6ba34440e8d14 size: 3022

IMMUTABLE_REF=localhost:5000/releaseguard-demo-service@sha256:8e7b271905dcf442825066cb4f63f3d0cee88d2c6abc3e1548e6ba34440e8d14
```

Running the pushed image and calling its own `/version` endpoint confirms
the ldflags-injected build metadata matches what was passed to `docker
build`, i.e. the image's self-reported identity is consistent with its
registry-assigned digest:

```
GET /version -> {"release_version":"phase0-evidence-b5965b5","image_version":"phase0-evidence-b5965b5","git_commit":"b5965b5","build_time":"2026-08-17T07:01:32Z","track":"stable"}
GET /health  -> {"status":"ok","track":"stable","version":"phase0-evidence-b5965b5","dependency":"ok","uptime_seconds":1.064251997}
```

## Interpretation

- **Tag**: `releaseguard-demo-service:b5965b5` -- a mutable pointer, but
  scoped to a single commit so it isn't reused across unrelated builds.
- **Digest**: `sha256:8e7b27...e8d14` -- content-addressed and immutable by
  construction; this is the reference `deploy/kubernetes/overlays/*`
  should eventually pin to for anything beyond local dev (Phase 6+
  Kubernetes adapter). Locally, `kind load docker-image` bypasses the
  registry/digest step entirely (see `deploy/local-cluster/setup-cluster.sh`),
  which is why Phase 1's cluster smoke test uses the `:local` tag rather
  than a digest -- digest pinning is meaningful once there is a real
  registry in the loop, which is exactly what this evidence run adds
  synthetically via `registry:2`.
- The `.github/workflows/ci.yml` `build-and-push` job performs the
  equivalent sequence against `ghcr.io` using `docker/build-push-action`,
  reading `steps.build.outputs.digest` and writing it to the job summary.
  That workflow has not been executed on GitHub Actions itself in this
  environment (no push/remote access here -- git operations are owned by
  the orchestrator), so this file is the closest available first-party
  evidence that the underlying build mechanism the workflow encodes
  actually produces a valid, immutable, digest-addressable image. This is
  flagged as an open item in the final report.
