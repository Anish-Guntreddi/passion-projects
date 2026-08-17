# Phase 0 FR1 evidence: image vulnerability scan

**What this file is:** a raw, measured record of running the same Trivy
scan `.github/workflows/ci.yml`'s `build-and-push` job runs, against the
demo-service image actually built by `docker build` from this repo's
`Dockerfile`. This is the evidence that the scan gate FR1 requires ("image
vulnerability/dependency scan") both exists and currently passes -- not
just that the YAML step is present.

## Environment

| | |
|---|---|
| Date (UTC) | 2026-08-17 |
| Host | Windows 11 Pro, WSL2 Ubuntu 24.04.1 LTS |
| Docker | 27.4.1 |
| Trivy | `aquasec/trivy:0.36.0` (same version pinned in `ci.yml`'s `aquasecurity/trivy-action@0.36.0`) |
| Trivy vulnerability DB | downloaded fresh at scan time (2026-08-17) |
| Image scanned | `releaseguard/demo-service:local`, built from this repo's `demo-service/Dockerfile` (`gcr.io/distroless/static-debian12:nonroot` base) |

## Command

```bash
docker build -t releaseguard/demo-service:local demo-service

docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.36.0 image \
  --format table --exit-code 1 --vuln-type os,library --severity CRITICAL,HIGH --ignore-unfixed \
  releaseguard/demo-service:local
```

These are the exact flags the CI step passes to `aquasecurity/trivy-action`
(`exit-code: "1"`, `vuln-type: "os,library"`, `severity: "CRITICAL,HIGH"`,
`ignore-unfixed: true`).

## Measured result

```
releaseguard/demo-service:local (debian 12.15)
==============================================
Total: 0 (HIGH: 0, CRITICAL: 0)

EXIT_CODE=0
```

## Interpretation

- Zero fixable CRITICAL/HIGH findings in either the Debian OS packages or
  the Go module dependencies baked into the distroless runtime image, as
  of this scan date.
- Exit code 0 confirms the gate, as configured, does not currently block
  CI -- this is expected to be re-measured whenever the base image or Go
  dependencies are updated, and the CI job itself re-runs this same check
  on every push/PR going forward (not just at this one point in time).
