# ADR-0012: Page-fault telemetry counters are lock-free, owner-only -- p->lock inside vmfault() is rejected

**Status:** Accepted; implemented in Phase 5
**Decision:** refines D6 (spec Â§1.10) for the Phase 5 fields, the same
way ADR-0009 refined it for Phase 2's fields
**Date:** 2026-08-17

## Context

`vmfault()` (`kernel/vm.c`) always operates on `struct proc *p =
myproc()` -- it services a fault on behalf of whichever process is
currently running on the calling hart. Phase 5 adds two counters,
`p->pagefaults` and `p->pagefaults_failed`, incremented at each of
`vmfault()`'s three outcomes (out-of-bounds address, `kalloc()`/
`mappages()` failure, success). Per ADR-0006's case (b), a field read
cross-process (both counters are read by `procstat()` for `xvstat(2)`,
just like Phase 2's `runticks`/`waitticks`/`syscall_count`) should
take the target process's own `p->lock` on write. This ADR is about
why that default was checked against this specific call site and
rejected, not applied automatically.

## The hazard

`vmfault()` is called from three places: `kernel/trap.c`'s
`usertrap()` (a genuine user page fault), and `kernel/vm.c`'s
`copyout()`/`copyin()` (a kernel-side lazy fault while servicing a
syscall argument). The second case matters here: `kernel/proc.c`'s
`kwait()` calls

```c
acquire(&pp->lock);
...
if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate, sizeof(pp->xstate)) < 0) {
```

-- i.e. it calls `copyout()` on the **waiting parent's** (`p`) own
page table, targeting the parent's own `wait()` buffer, **while still
holding the reaped child's (`pp`) `p->lock`**. If that `addr` happens
to fall in a not-yet-touched lazily-`sbrk`'d region of the parent's
own memory (a legal, if unusual, choice of buffer -- `wait()` takes
any writable user pointer), `copyout()` calls `vmfault(p->pagetable,
...)`, which runs as `myproc() == p`, the parent. If `vmfault()`
itself acquired `p`'s own `p->lock` to update the new counters, this
call path would then be holding **two different processes' `p->lock`
at once** (`pp`'s, acquired by `kwait()`, then `p`'s, acquired inside
`vmfault()`) -- a lock-order edge (`pp->lock -> p->lock`, for an
arbitrary parent/child pair) that does not exist anywhere else in this
codebase. ADR-0009's own rationale explicitly relies on "no path
acquires two different processes' `p->lock` at once" holding
codebase-wide; introducing this edge silently for a monitoring-only
telemetry feature would be exactly the kind of unreviewed lock-order
change invariant #2 exists to catch.

(No path today acquires some process's `p->lock` and *then* tries to
acquire a second, different process's `pp->lock` in the reverse
nesting, so this specific edge is very unlikely to deadlock in
practice -- but "unlikely to deadlock today" is not the same as "not a
new lock-order edge," and per-kernel-task requirement discipline
(handoff brief) means treating it as one rather than shipping it
unexamined.)

## Decision

`vmfault()`'s counter increments (`p->pagefaults++` /
`p->pagefaults_failed++`) are **not** protected by `p->lock`. They
follow the same lock-free "owner-only mutation" convention ADR-0006
case (a) established for `trace_mask`: only the one hart currently
running `myproc() == p` ever touches `p`'s own counters at this code
point (a process cannot be concurrently executing on two harts at
once), so there is no concurrent-*writer* race to protect against --
only a concurrent *reader* (`xvstat(2)` from another process), which
gets the same "naturally-aligned 64-bit load/store is never torn, only
possibly stale" guarantee ADR-0009 already accepted for `sz` and
`name` in the very same `struct xv_pstat`. `procstat()` reads
`pagefaults`/`pagefaults_failed` in the same critical section as the
hard-guaranteed fields, but documents (in its own comment and in
`kernel/proc.h`'s field comment) that these two specifically do not
carry that hard guarantee -- exactly the `sz`/`name` precedent, not a
new one.

## Rationale

This is not "skip locking because it's easier" -- it is the direct,
documented consequence of a real lock-order hazard found by tracing
every call site of `vmfault()`, the same rigor ADR-0009 applied when
it rejected reusing `tickslock` for the `waitticks` write. Taking
`p->lock` here would fix nothing (the counters are already race-free
against concurrent writers by construction) while adding a new,
previously-nonexistent cross-process lock nesting for no benefit.

## Consequences

A future VM feature that wants a *hard* consistency guarantee on a
per-process counter must re-derive this analysis for its own mutation
site, not assume "cross-process-read implies take p->lock" is a
context-free rule -- ADR-0006 already said as much for D6 generally
("any future accounting counter... breaks the 'owner-only mutation'
assumption... needs its own ADR"); this is the second time (after
ADR-0009) that escape hatch has been exercised, and both times the
answer depended on tracing actual call sites, not applying the
general rule by default. `docs/vm-extension.md` and the field comments
in `kernel/proc.h`/`kernel/pstat.h` cross-reference this ADR directly.
