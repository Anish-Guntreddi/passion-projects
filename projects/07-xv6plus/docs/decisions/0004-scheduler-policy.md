# ADR-0004: Scheduler experiment is lottery scheduling

**Status:** Accepted; implemented in Phase 4. See
[ADR-0010](0010-lottery-scheduler-design.md) for the implementation
decisions (PRNG, two-pass draw, tickets lifecycle, zero-ticket floor)
this ADR deferred below.
**Decision:** D4 (spec §1.10)
**Date:** 2026-08-17 (recorded ahead of implementation; see ADR-0003)

## Context

FR5 / Phase 4 requires one scheduler extension beyond the baseline
round-robin scheduler, with policy selection and documented
fairness/starvation behavior. D4's spec default: "lottery scheduling
(clean fairness story)."

## Decision

Adopt the spec default: lottery scheduling as the Phase 4 scheduler
experiment, selectable alongside the existing baseline round-robin
`scheduler()` (`kernel/proc.c`) via a policy switch, not a wholesale
replacement of it. Not implemented yet; recorded now, per the kickoff
prompt, so later phases build toward one settled direction and Phase
1's telemetry design doesn't accidentally foreclose it.

## Rationale

Lottery scheduling gives a fairness story that is easy to reason about
and to benchmark: ticket counts map directly to an expected CPU share,
which is directly testable against measured selection counts (spec
§1.9: "selection counts, fairness distribution"). It is a well-scoped
change relative to xv6's existing round-robin loop, keeping it
interview-explainable (handoff rule 5).

## Consequences

Needs a per-process "tickets" field with the same lifecycle discipline
already established for `trace_mask` in Phase 1: initialized at
allocation, explicit fork/exec semantics decided in Phase 4, cleared
on free (invariants #3, #5). Needs a PRNG seeded in a documented,
reproducible way so any measured fairness result can be reproduced
exactly, per the portfolio's no-fabricated-numbers rule. Starvation
risk (a process with zero or near-zero tickets) must be explicitly
bounded and tested against invariant #4 ("scheduler always eventually
selects eligible work").
