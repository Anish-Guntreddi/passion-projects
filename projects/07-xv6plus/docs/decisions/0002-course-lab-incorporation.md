# ADR-0002: Zero course lab solutions incorporated

**Status:** Accepted (implementer decision, see Context)
**Decision:** D2 (spec §1.10)
**Date:** 2026-08-17

## Context

D2 ("Which standard labs are already incorporated") is marked a human
decision in the spec, documented in `docs/upstream-delta.md`. As with
ADR-0001, no human was available mid-build to state which labs (if
any) had already been completed elsewhere.

## Decision

Assume, and verify, that **zero** standard course lab solutions are
incorporated into this repository. The pinned base (ADR-0001) is
`mit-pdos/xv6-riscv`'s own main-line source -- the same reference
kernel every MIT 6.1810 lab is handed out as a diff against -- not the
separate `xv6-labs-*` per-lab-branch repository. `kernel/` and `user/`
were inspected directly after vendoring and contain no lab-specific
scaffolding (no lazy-allocation TODO stubs, no COW hint comments, no
mmap stubs, etc.).

## Rationale

This is the safest, most honest default under the project's own
load-bearing honesty requirement (§1.2): claiming lab credit that
cannot be independently verified by a reviewer is exactly the failure
mode §1.2 exists to prevent. Starting from zero means every feature in
`docs/upstream-delta.md` category (C) is provably original xv6-plus
work, satisfying handoff rule 2 ("Do not copy unseen lab solutions as
implementation shortcuts") by construction rather than by policy.

## Consequences

`docs/upstream-delta.md` category (B) ("completed educational lab
modifications") is empty for this repository; everything beyond
pristine upstream is category (C). If the human author has, in fact,
separately completed standard labs and wants them credited, that is a
deliberate follow-up decision (e.g. re-basing onto a specific labs
branch with clear attribution), not something this ADR does silently.
