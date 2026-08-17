# ADR-0011: VM extension choice finalized -- page-fault telemetry

**Status:** Accepted; implemented in Phase 5 (supersedes ADR-0005's deferral)
**Decision:** D5 (spec Â§1.10)
**Date:** 2026-08-17

## Context

ADR-0005 deliberately deferred D5 rather than guessing, and listed two
standing candidates in preference order: (1) an `mmap()` subset, (2)
lazy allocation + page-fault telemetry, noting that this pinned
upstream revision's `sys_sbrk()` already has an eager/lazy split
(`kernel/vm.h`: `SBRK_EAGER`/`SBRK_LAZY`) with lazy growth deferring to
an existing `vmfault()` fault-handling seam (`kernel/vm.c`,
`kernel/trap.c`'s `usertrap()`). As with D1/D2 (ADR-0001/0002), no
human was available mid-build to make the requested human decision, so
the implementer picks the most defensible, objective option rather
than blocking.

## Decision

**Page-fault telemetry** -- explicitly one of FR6's own listed
options ("mmap subset, lazy allocation, COW-related instrumentation,
accessed-page stats, **or page-fault telemetry**"), building on the
existing (already-upstream, per ADR-0002's honesty-layer inspection)
lazy-allocation fault path rather than introducing an entirely new
`mmap()` syscall and VMA bookkeeping structure.

## Rationale

- **Lower risk, same invariant coverage.** ADR-0005 already reasoned
  that option 2 is the lower-risk choice because "the fault-handling
  seam this extension would grow from already exists upstream." An
  `mmap()` subset would be a substantially larger new surface (a new
  syscall, new per-process VMA bookkeeping, new interactions with
  `fork()`/`exec()`/`exit()` cleanup) for a single implementer to get
  right without human review mid-build -- exactly the kind of
  "premature, unreviewed architectural commitment" ADR-0005 itself
  warned against for D5 as a whole.
- **Directly named by FR6.** Unlike improvising a variant not on the
  spec's own list, page-fault telemetry is explicitly enumerated as an
  acceptable Phase 5 deliverable, so this choice needs no additional
  justification for scope.
- **Untested surface, not just new code.** `docs/invariants.md`'s
  Phase 3 status table marks invariant #6 (VM isolation/mapping
  permissions) "Not applicable: Phases 1-3 make no VM changes" --
  meaning the existing `vmfault()` path, despite already being live in
  every process's page-fault handling since before this project even
  began, had **zero dedicated tests**. A telemetry-plus-hardening pass
  that adds real correctness tests for an already-shipping mechanism
  is a genuine, substantive contribution even though the underlying
  fault-in mechanism itself is upstream, not original -- see
  `docs/vm-extension.md` for exactly what is upstream (A) vs. original
  xv6-plus (C) in this feature, per the spec Â§1.2 honesty requirement.

## Consequences

`docs/upstream-delta.md`'s Phase 5 entry must be explicit that the
underlying lazy-fault mechanism (`vmfault()`, `sys_sbrk()`'s
`SBRK_LAZY` path) is category (A), unmodified in its actual allocation
logic -- only the telemetry counters, the `kernel/trap.c` diagnostic
split, and the dedicated test suite are category (C). This is a
deliberately narrower originality claim than "implemented lazy
allocation" would be, and is the honest one. ADR-0005 remains as the
historical record of the deferral and the candidate list this ADR
chose from; its own "Consequences" section already anticipated this
exact outcome ("prefer option 2... since it builds on a fault path
this pinned revision already partially has").
