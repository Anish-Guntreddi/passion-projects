# Core invariants

The eight invariants from spec §1.5, each guarded by tests/review, and
their status as of Phase 1. This document is updated every phase; a
stale entry here is a bug in the process, not just in the docs.

| # | Invariant | Status after Phase 1 |
|---|---|---|
| 1 | Kernel accounting never corrupts process lifecycle state. | Partially exercised: no accounting *counters* exist yet (Phase 2), but the `trace_mask` field added in Phase 1 follows the exact struct-proc lifecycle path Phase 2's counters will reuse, and `tests/syscall/test_trace_fork_inheritance.py` proves `forktest`'s full proc-table-filling fork/reap cycle completes correctly (`"fork test OK"`) with that new field live on every process. |
| 2 | Locks are never acquired in inconsistent order. | Not at risk in Phase 1: no new lock was introduced (ADR-0006); `wait_lock` -> `p->lock` ordering, documented at the top of `kernel/proc.c`, is unchanged. |
| 3 | Process telemetry is initialized at allocation and handled correctly on fork/exec/exit. | **Directly implemented and tested.** `allocproc()` and `freeproc()` both zero `trace_mask` (`kernel/proc.c`); `kfork()` copies it parent-to-child; `kexec()` is untouched, so it survives exec by construction. Tested by `test_trace_fork_inheritance.py` (survives many forks) and `test_trace_toggle.py` (survives within one process's lifetime). |
| 4 | The scheduler always eventually selects eligible work per documented policy assumptions. | Not modified in Phase 1 (Phase 4). Indirectly exercised: `test_trace_fork_inheritance.py`'s `forktest` run needs the baseline round-robin scheduler to keep making progress across dozens of fork/exit cycles with tracing active on every process, and it does. |
| 5 | Zombie/free process slots never remain visible as active telemetry. | **Directly implemented and tested.** `freeproc()` resets `trace_mask` to 0 when a slot returns to `UNUSED`. `test_trace_isolation.py` proves a freshly forked process (reusing a just-freed slot, since `NPROC` is finite) never inherits a previous occupant's trace state. |
| 6 | VM changes preserve user/kernel isolation and mapping permissions. | Not applicable: Phase 1 makes no VM changes (deferred to Phase 5, ADR-0005). |
| 7 | Syscall interfaces validate user pointers/arguments. | `trace(int mask)` takes only a plain `int` (via `argint()`); there is no user pointer for this syscall to validate, by design (see `docs/tracing.md`, "safe scope"). This becomes directly relevant again in Phase 2, when a stats syscall `copyout()`s a struct to a user buffer. |
| 8 | Observability code must not destabilize the kernel it observes. | **Directly tested.** `test_no_regression.py` proves a session where `trace()` is never called behaves exactly like a stock boot (no trace lines leak in, no `"unknown sys call"`, no panic). `test_trace_fork_inheritance.py` proves heavy simultaneous tracing + proc-table-filling load doesn't destabilize anything either. |

## Known, deliberate limitation

`SYS_exit` can never produce a trace line: `syscall()` in
`kernel/syscall.c` prints the trace line *after* calling the
dispatched syscall handler, but `sys_exit()` -> `kexit()` never
returns to that call site -- it jumps directly into `sched()` and the
process is gone. This is not a bug to fix; it is a direct structural
consequence of how xv6 implements process exit, documented here so it
isn't mistaken for an oversight, and it does not violate invariant #8
(nothing unsafe is attempted -- the print statement is simply never
reached).
