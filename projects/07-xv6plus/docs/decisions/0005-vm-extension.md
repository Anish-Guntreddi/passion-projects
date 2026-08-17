# ADR-0005: VM extension choice -- deferred

**Status:** Open (deferred to Phase 5, human input requested)
**Decision:** D5 (spec §1.10)
**Date:** 2026-08-17

## Context

FR6 / Phase 5 requires one bounded VM extension: mmap subset, lazy
allocation, COW instrumentation, accessed-page stats, or page-fault
telemetry. The spec is explicit that D5 is "decided after prerequisite
labs" and needs human input -- unlike D3/D4/D6/D7, it states no
recommended default.

## Decision

**Not decided now.** This ADR exists only to record the deferral
explicitly and list the spec's own standing candidates, so Phase 5
planning starts from an ordered short list instead of a blank page:

1. **mmap subset** (spec's stated preference #1) -- touches `uvm*` in
   `kernel/vm.c` and the fault path in `kernel/trap.c`'s
   `usertrap()`; exercises invariant #6 (VM isolation/mapping
   permissions) most directly.
2. **Lazy allocation + page-fault telemetry** (spec's stated
   preference #2) -- smaller, more self-contained; this pinned
   revision's `sys_sbrk()` (`kernel/sysproc.c`) already distinguishes
   an eager path from a lazy path (`SBRK_EAGER` vs. `SBRK_LAZY`,
   defined in `kernel/vm.h`), with lazy growth deferring to
   `vmfault()` (`kernel/defs.h`, implemented in `kernel/vm.c`), so the
   fault-handling seam this extension would grow from already exists
   upstream.

## Rationale for not guessing

Unlike D1/D2 (a defensible conservative default exists) or D3/D4/D6/D7
(the spec states its own default), D5 has none -- the spec says this
choice should follow from what Phases 1-3 actually teach about this
codebase's page-table and fault-handling code, none of which has been
built yet in this change. Picking now would be exactly the kind of
premature, unreviewed architectural commitment ADRs exist to prevent.

## Consequences

Revisit at Phase 5 kickoff. Ideally with human input per the spec's
own instruction; absent that, prefer option 2 (lazy allocation +
fault telemetry) as the lower-risk choice, since it builds on a fault
path this pinned revision already partially has, rather than adding
an entirely new `mmap()` syscall and VMA bookkeeping structure.
