# ADR 0002: Deployment manifests use Kustomize, not Helm

**Status:** Accepted
**Decision ID:** tech-stack table row "Manifests" (spec Part 2)
**Date:** 2026-08-17

## Context

FR2 and the tech-stack table require choosing **one** primary manifest
approach -- Helm or Kustomize -- and documenting why. This project's local
Kubernetes environment (kind) is meant to be reproducible with minimal
external tooling (spec §1.4 non-goals rule out being a general deployment
framework; Claude Code handoff rule 3: "keep the local MVP independent of
paid cloud services," which by extension favors fewer moving parts overall).

## Decision

Use **Kustomize**, applied via `kubectl apply -k` (kustomize is built into
kubectl since 1.14, so no extra binary is required beyond kubectl itself,
which the cluster scripts already depend on).

Layout:

```
deploy/kubernetes/
  base/                   # plain Kubernetes YAML: Namespace, Deployment, Service
    kustomization.yaml
    namespace.yaml
    deployment.yaml
    service.yaml
  overlays/
    local/                # kind-specific overrides (image tag, replica count, etc.)
      kustomization.yaml
```

This folds the spec's repo-structure sketch (`deploy/{kubernetes,
helm-or-kustomize, local-cluster}/`) into two real directories:
`deploy/kubernetes/` holds both the base manifests *and* the kustomize
overlays (since kustomize operates directly on plain Kubernetes YAML,
there is no separate templating artifact the way a Helm chart would
require), and `deploy/local-cluster/` holds kind-cluster lifecycle scripts.
The literal `helm-or-kustomize/` directory name from the spec sketch is not
created as-is; it was a placeholder for "whichever tool you pick," and
Kustomize does not need a directory separate from the manifests it
overlays.

## Consequences

- No Helm binary to install, version-pin, or explain in the README;
  `kubectl` alone is sufficient for Phase 1's "fresh clone can deploy
  locally" exit criterion.
- Overlays (`overlays/local`, and later `overlays/canary` in Phase 3) are
  plain strategic-merge patches over real Kubernetes YAML, which is easier
  to read and diff in a portfolio review than a templated `values.yaml` +
  Go-template chart.
- Helm's templating power (conditionals, loops, a packaged/versioned
  chart artifact) is not needed at this MVP's scale (one Deployment + one
  Service per track, plus a Namespace) and is left off the table
  intentionally rather than by omission.

## Alternatives considered

- **Helm**: the more common choice for larger real-world deployments and
  what most operators expect, but it requires installing and pinning a
  separate binary, has a steeper "why did templating produce this YAML"
  debugging story, and its main value (packaging/distribution, values
  overrides across many environments) is not exercised by a single-cluster
  local MVP. Revisit if/when this project grows multi-environment
  (cloud) deployment (D7, deferred).
