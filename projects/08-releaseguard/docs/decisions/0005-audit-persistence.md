# ADR 0005: Audit trail persisted as append-only JSONL

**Status:** Accepted (implementation lands in Phase 5, not this change)
**Decision ID:** D4 (spec §1.9)
**Date:** 2026-08-17

## Context

D4 asks how the deployment audit log (FR4: "answers *why did ReleaseGuard
make this decision?*") should be persisted, defaulting to "append-only
JSONL file (simplest robust option); SQLite or K8s CR/status as
alternatives."

## Decision

Persist every audit event (state transition, decision, evidence snapshot)
as one JSON object per line, appended to a file
(`controller/internal/audit`, path configurable, defaulting to something
like `/var/lib/releaseguard/audit.jsonl` in-cluster or a local path for
dev). Each line is self-describing (event type, timestamp, rollout ID,
reason codes, evidence references) so the file can be tailed, grepped, or
loaded into any JSON-line-aware tool without a schema migration step.

## Consequences

- Append-only writes are simple to make crash-safe (open in append mode,
  `fsync` after each write, no in-place mutation) which directly serves
  §1.7's reliability requirement ("controller restart during observation/
  action" must not corrupt or duplicate audit state) and Phase 9's
  hardening goals.
- Human- and grep-friendly by construction -- a core deliverable is
  "audit-log decision explanation" for the portfolio website (§1.5); a
  JSONL file is trivial to pretty-print into a timeline view or feed to a
  small script that renders the incident-demo narrative (docs/
  incident-demo.md, Phase 10).
- No query language, indexing, or concurrent-writer story beyond "single
  controller process, sequential appends" -- acceptable because the MVP
  is explicitly a single standalone controller service (ADR 0006), not a
  multi-replica control plane.
- Querying "all decisions for rollout X" requires a linear scan (or an
  external `jq`/log-shipping step) rather than an indexed lookup; deferred
  as acceptable for MVP scale and revisited only if Phase 9 hardening
  finds it insufficient.

## Alternatives considered

- **SQLite**: gives indexed queries and transactional writes for free,
  and remains a zero-external-dependency single file, making it a
  reasonable upgrade path later. Not chosen initially because it adds a
  schema/migration surface for no MVP-scale benefit, and JSONL is more
  directly "evidence you can read," which matters for a project whose
  point is auditability.
- **Kubernetes CR/status subresource**: would make the audit trail
  queryable via `kubectl get releaseguardrollout` and fit naturally with
  a CRD/operator pattern, but that pattern itself is explicitly deferred
  (ADR 0006 / D5: standalone service for MVP, CRD/operator is stretch).
  Revisit together with D5 if the operator pattern is adopted later.
