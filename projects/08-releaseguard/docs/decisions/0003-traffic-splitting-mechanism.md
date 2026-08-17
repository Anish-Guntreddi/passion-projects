# ADR 0003: Traffic-splitting mechanism is replica-count-based

**Status:** Accepted (implementation lands in Phase 3, not this change)
**Decision ID:** D2 (spec §1.9)
**Date:** 2026-08-17

## Context

D2 asks how canary traffic splitting should work while staying lightweight
locally, with a recommended default of "replica-count-based split or
deterministic client-side sampling."

This decision is recorded now (ahead of Phase 3, "Manual canary") per the
spec's kickoff prompt ("ADR decisions D1-D6 before their phases"), even
though Phase 0/1 only deploy a single `stable` track and do not yet need a
canary workload.

## Decision

Split traffic between `stable` and `canary` by **replica count** behind a
single Kubernetes Service that selects both tracks via a shared label
(e.g. `app.kubernetes.io/name: demo-service`, with `release.releaseguard/
track: stable|canary` distinguishing pods). Kubernetes Services load-balance
across all matching endpoints roughly uniformly (via kube-proxy iptables/
IPVS round-robin), so a 9:1 stable:canary replica ratio yields an
approximate 90/10 traffic split without any additional traffic-management
component (no service mesh, no ingress-level weighted routing).

This is deliberately coarse-grained (ratios are quantized by replica
count -- a 1:1 split from `replicas: 1` each is the finest-grained useful
canary ratio without adding more moving parts) and is exactly why FR3
requires labeling every metric with `track`/`version`, so the evaluator
(Phase 4) measures per-track behavior directly from Prometheus rather than
depending on the split ratio being exact.

## Consequences

- No service mesh (Istio/Linkerd), no ingress controller, no extra CRDs --
  matches Claude Code handoff rule 3 (independent of paid/heavy cloud
  infrastructure) and spec §1.4 non-goals (no service-mesh traffic control
  in MVP; that is an explicit stretch goal).
- The achievable split granularity is limited by replica count (e.g. with
  3 canary + 7 stable replicas the split is ~30/70, not exactly 30/70,
  because kube-proxy balances per-connection/per-request approximately,
  not by a precise percentage). This is acceptable for MVP because the
  evaluator's minimum-sample-count guardrail (Phase 4, FR4) is what
  actually decides whether there is enough canary traffic to trust a
  decision -- the split mechanism only needs to produce *some* canary
  traffic, not an exact percentage.
- Scaling stable/canary replica counts is the one action the Phase 6
  Kubernetes adapter needs to perform for promote/rollback, which keeps
  that adapter's surface area small (patch replica counts + swap which
  track is "primary"), directly serving Claude Code handoff rule 2
  ("make all rollout actions idempotent" -- a replica-count patch is
  naturally idempotent: applying `replicas: N` twice is a no-op).

## Alternatives considered

- **Deterministic client-side sampling** (e.g. hash the request/session
  ID and route a fixed percentage to canary): gives an exact, stable
  percentage independent of replica count, but requires either a sidecar/
  proxy or application-level routing logic that this MVP's demo-service
  does not have (and adding it would blur the "demo app is deliberately
  simple" boundary from spec §1.2).
- **Service mesh weighted routing** (Istio VirtualService, Linkerd
  TrafficSplit): the most precise and production-realistic option, but
  explicitly listed as a stretch goal, not MVP, and adds substantial
  local-cluster complexity (extra control-plane components, sidecar
  injection) that Phase 1's exit criterion ("fresh clone can deploy
  locally") should not have to depend on.
