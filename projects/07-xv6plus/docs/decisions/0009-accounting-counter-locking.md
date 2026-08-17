# ADR-0009: Accounting counters take p->lock explicitly; tickslock reuse is rejected

**Status:** Accepted; implemented in Phase 2
**Decision:** refines D6 (spec §1.10) -- the specific cross-process-read
case ADR-0006 reserved a follow-up ADR for
**Date:** 2026-08-17

## Context

ADR-0006 set the general telemetry-synchronization rule: (a) a field
mutated only by its own owning process needs no lock beyond xv6's
existing "private to the process" convention (the `trace_mask`
precedent); (b) a field read by any *other* process must be read
under that target process's existing `p->lock`, same as
`kkill()`/`kwait()`/`procdump()` already do. ADR-0006 explicitly
deferred settling case (b) in detail until a real cross-process-read
field existed.

Phase 2 adds exactly that: `runticks`, `waitticks`, and
`syscall_count` (`kernel/proc.h`) are each written by kernel code
running *as* the owning process, but read cross-process by
`procstat()`/`xvstat(2)` (ADR-0003) enumerating the whole proc table
for `xvtop` and the accounting test programs. This ADR settles how
each counter's write side is locked to match, and records a specific
lock-order hazard that came up while doing so.

## Decision

Every write to `runticks`, `waitticks`, or `syscall_count` takes the
target process's own `p->lock`, per ADR-0006 case (b):

- **`runticks`** -- `kernel/trap.c`'s `clockintr()`, which runs on
  every hart's own periodic timer interrupt, acquires `myproc()->lock`
  around `p->runticks++` if a process is currently running on that
  hart. `clockintr()` is only ever reached with no spinlock already
  held on that hart (taking any spinlock disables interrupts for its
  duration, which is exactly what would have prevented this timer trap
  from being taken at all), so this acquire cannot nest under anything
  already held.
- **`waitticks`** -- `kernel/proc.c`'s `sleep()` already holds
  `p->lock` for its own state-change duties; it adds
  `(ticks - sleep_start)` to `p->waitticks` in that same critical
  section, no extra acquire needed.
- **`syscall_count`** -- `kernel/syscall.c`'s `syscall()`, the
  per-trap dispatcher, acquires and releases `p->lock` around
  `p->syscall_count++` *before* dispatching to the syscall handler, so
  the handler is free to acquire `p->lock` again itself (e.g.
  `sys_xvstat()` -> `procstat()`) without deadlocking against a lock
  this function is still holding.

`procstat()` (`kernel/proc.c`), the single reader behind `xvstat(2)`,
acquires proc `idx`'s `p->lock` once and reads `pid`, `state`,
`runticks`, `waitticks`, and `syscall_count` together in that critical
section, giving those five fields a hard, mutually-consistent
snapshot. `sz` and `name` are read in the same critical section but do
**not** carry that guarantee: their existing writers (`growproc()`,
`kexec()`) intentionally keep the pre-existing ADR-0006-case-(a)
"private to the process" convention and were not changed for this
feature, so a concurrent grow/exec on the target can still be observed
mid-update. Accepted as monitoring-only staleness -- invariant #8 does
not require a safety property here, only that observability code not
destabilize the kernel, and a naturally-aligned 64-bit RISC-V
load/store can never produce a torn value, only a momentarily stale
one.

**`tickslock` reuse was considered and rejected** for `waitticks`
specifically. `clockintr()` already holds `tickslock` elsewhere in the
same function (to increment the global `ticks` and call
`wakeup(&ticks)`), and `wakeup()` acquires `p->lock` for every
sleeping process it scans -- establishing the lock order
`tickslock -> p->lock`. `sleep()` runs in the opposite direction: by
the point it would want to read the tick count for `waitticks`
accounting, it already holds `p->lock` for its own state change.
Taking `tickslock` there too would be `p->lock -> tickslock`, the
reverse order, and a two-hart deadlock is the textbook consequence of
two locks acquired in opposite orders on different call paths. Two
ways out existed: (1) read the plain `ticks` global unlocked instead
of taking `tickslock` in `sleep()`, or (2) introduce a new lock.
Option (1) was chosen: RISC-V guarantees a naturally-aligned load of a
plain word observes only a fully-formed old or new value, never a
torn one, which is all a best-effort accounting counter needs
(ADR-0007 already accepts tick-granularity, not cycle-accurate,
precision) -- and it avoids adding a fourth lock to a kernel whose
entire lock order is otherwise documented in one place (the top of
`kernel/proc.c`).

## Rationale

Reusing `p->lock` keeps every Phase 2 counter inside the exact
lock-order graph ADR-0006 and the top of `kernel/proc.c` already
document (`wait_lock -> p->lock`; no path acquires two different
processes' `p->lock` at once). Rejecting `tickslock` reuse for the
`sleep()`/`waitticks` write avoids adding a new cross-lock edge that
upstream xv6's own `clockintr() -> wakeup() -> p->lock` chain would
otherwise turn into a deadlock -- exactly the kind of lock-order
inconsistency invariant #2 exists to rule out.

## Consequences

`pid`, `state`, `runticks`, `waitticks`, and `syscall_count` carry a
hard, mutually-consistent read guarantee from `xvstat(2)`; `sz` and
`name` do not, and that gap is accepted rather than fixed (documented
at the `procstat()` call site and in `docs/accounting.md`), since
fixing it would mean changing `sz`/`name`'s existing, upstream-adjacent
locking convention for a monitoring-only feature. Any future
accounting field must be run through this same question -- "is it
only ever read by its own process, or could some other process's read
need a consistent snapshot?" -- before defaulting to ADR-0006 case (a)
or this ADR's case (b); a field this reasoning doesn't already cover
should get its own follow-up note here rather than silently assuming
one of the two existing patterns applies.
