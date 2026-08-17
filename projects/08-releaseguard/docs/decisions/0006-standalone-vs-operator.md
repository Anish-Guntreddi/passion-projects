# ADR 0006: Standalone controller service, not a CRD/operator, for MVP

**Status:** Accepted
**Decision ID:** D5 (spec §1.9)
**Date:** 2026-08-17

## Context

D5 asks whether ReleaseGuard should be a standalone controller service or
a Kubernetes CRD/operator, defaulting to "standalone controller service
for MVP; CRD/operator is stretch." This decision shapes the repo layout
being scaffolded now (`controller/cmd/`, no `config/crd/` or
controller-runtime scaffolding), so it is recorded in this change even
though the controller itself is built starting Phase 4.

## Decision

ReleaseGuard's controller is a standalone Go binary/service
(`controller/cmd/`) that runs its evaluation loop against the Kubernetes
API and Prometheus using plain client libraries (`client-go`,
`prometheus/client_golang`), not a `controller-runtime`-based operator
watching a custom resource. Rollout policy is supplied as a config file
(YAML/JSON, FR4), not a CRD spec; decisions and audit records are exposed
via the JSONL audit log (ADR 0005) and, if a CLI/API is added, a
service-local endpoint -- not `kubectl get`.

## Consequences

- No CRD schema, no controller-runtime manager/reconciler boilerplate, no
  RBAC surface for arbitrary custom-resource CRUD -- keeps Phase 8's
  least-privilege RBAC review scoped to exactly what the controller
  touches (Deployments, Services, Pods for status; nothing else).
- Faster to build and interview-explain: a standalone service with an
  explicit reconciliation loop is easier to read start-to-finish than
  operator scaffolding, matching the portfolio-wide "code stays
  interview-explainable" constraint.
- Operators are the more idiomatic Kubernetes-native pattern for this kind
  of problem (Argo Rollouts and Flagger, the tools §1.4 explicitly says
  ReleaseGuard is not trying to replace, are both operators) -- adopting
  it later is the natural "stretch" evolution the spec calls out, and
  nothing in this MVP's design (config-driven policy, adapter-based
  Kubernetes actions) should need to be thrown away to get there; a CRD
  would mostly formalize the existing policy schema and audit events as
  a `Status` subresource.

## Alternatives considered

- **CRD/operator from the start**: rejected for MVP scope per the spec's
  explicit default and stretch-goal list; revisit post-MVP if the
  portfolio narrative benefits from demonstrating operator patterns
  (`controller-runtime`, CRD versioning, webhooks).
