# ADR 0004: Architecture-Safe Warp Primitives (D4)

- **Status:** Accepted (not yet exercised)
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D4 — default: `__shfl_sync`
  family with full masks.

## Context
Phases 0–1 do not use any warp-level shuffle/vote primitives (vector add,
SAXPY, transpose, and the stride microbenchmark are all thread-independent
memory operations). This decision is recorded now, ahead of Phase 2
(reduction/scan), so the convention is fixed before any warp-level code is
written, per the spec's instruction to resolve D1–D8 before Phase 0.

## Decision
Whenever warp-level primitives are introduced (first expected in Phase 2
reductions), the codebase uses only the **`_sync` family**
(`__shfl_sync`, `__shfl_down_sync`, `__ballot_sync`, `__any_sync`,
`__all_sync`, …) with an **explicit, full participation mask**
(`0xffffffffu` for warp-uniform control flow, or an explicitly computed
mask when a kernel has divergent participation). The legacy non-`_sync`
intrinsics (`__shfl`, `__ballot`, …) are never used — they are undefined
behavior on independent-thread-scheduling architectures (Volta+) and this
project's minimum capability (ADR 0002, sm_89) postdates that change by
several generations.

## Consequences
- No action required in Phases 0–1.
- Phase 2+ kernel code review checklist gains one item: every warp
  intrinsic call site must show its mask is either `0xffffffffu` under a
  warp-uniform branch, or derived from `__activemask()`/`__ballot_sync`
  with a comment justifying why that mask is correct at that call site.
