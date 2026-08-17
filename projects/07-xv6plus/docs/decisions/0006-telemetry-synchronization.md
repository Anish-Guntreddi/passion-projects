# ADR-0006: Telemetry synchronization -- owner-mutates, existing p->lock convention

**Status:** Accepted (spec default); validated in Phase 1, carries
forward to Phase 2
**Decision:** D6 (spec §1.10)
**Date:** 2026-08-17; revisited after Phase 1 implementation

## Context

Accounting fields (Phase 2) and the Phase 1 tracing control both need
a synchronization story. D6's spec default: "per-process counters
updated under existing proc lock," with an ADR called for only if new
locks turn out to be needed.

## Decision

Adopt the spec default, refined by what Phase 1 actually required:

`trace_mask` (`kernel/proc.h`) is mutated only by the owning process's
own kernel thread, inside its own syscall path (`sys_trace()` in
`kernel/sysproc.c`), and read only from that same process's own
syscall-dispatch code (`syscall()` in `kernel/syscall.c`). It follows
the exact "private to the process, `p->lock` need not be held"
convention xv6 already documents and uses for `sz`, `pagetable`,
`ofile`, `cwd`, `name` (see the comment block in `kernel/proc.h`). **No
new lock was introduced in Phase 1.**

This fixes the rule that will govern Phase 2 accounting fields too:

- (a) A field mutated only by its own owning process, during its own
  execution, needs no lock beyond that existing convention.
- (b) A field read by any *other* process (e.g. a future stats
  syscall enumerating all procs, per ADR-0003) must be read under
  that target process's existing `p->lock`, exactly as
  `kkill()`/`kwait()`/`procdump()` already do when touching another
  process's fields.

## Rationale

Reuses xv6's existing, already-reviewed locking discipline instead of
inventing a parallel one. This keeps the `wait_lock` -> `p->lock`
ordering (documented at the top of `kernel/proc.c`) unchanged, and
keeps the change interview-explainable: the answer to "why is
`trace_mask` safe without a lock?" is "for the same reason `p->sz`
is."

## Consequences

If a future accounting counter must be updated from *outside* the
owning process's own execution context (e.g. hypothetically, a
timer-interrupt handler on another hart incrementing a different
process's counter -- not currently designed, and not expected given
D7/ADR-0007 keeps counters at tick granularity updated by the process
itself at yield/sleep/exit points), that specific field breaks the
"owner-only mutation" assumption this ADR relies on and needs its own
ADR, per D6's own escape hatch.
