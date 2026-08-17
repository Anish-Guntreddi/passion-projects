# Core invariants

The eight invariants from spec §1.5, each guarded by tests/review, and
their status as of Phase 3. This document is updated every phase; a
stale entry here is a bug in the process, not just in the docs.

| # | Invariant | Status after Phase 3 |
|---|---|---|
| 1 | Kernel accounting never corrupts process lifecycle state. | **Directly implemented and tested.** Phase 2 adds three real accounting counters (`runticks`, `waitticks`, `syscall_count`) to `struct proc`, updated at `sleep()`/`clockintr()`/`syscall()` under `p->lock` (ADR-0009). Upstream's own `usertests` regression suite reports `ALL TESTS PASSED` with these fields and their locking live on every process, and `tests/accounting/test_slot_reuse_reset.py` / `test_fork_fresh_counters.py` prove the counters don't corrupt allocation/fork/free behavior. |
| 2 | Locks are never acquired in inconsistent order. | **Directly addressed.** Phase 2 introduces no new lock; every accounting-counter write reuses the existing `p->lock` (ADR-0006 case (b), ADR-0009). ADR-0009 specifically documents and avoids one hazard: reusing `tickslock` for the `waitticks` write in `sleep()` would establish `p->lock -> tickslock`, the reverse of `clockintr()`'s existing `tickslock -> p->lock` (via `wakeup()`) order -- rejected in favor of an unlocked read of the naturally-aligned `ticks` word. `wait_lock -> p->lock` ordering, documented at the top of `kernel/proc.c`, remains unchanged through Phase 3 (`xvtop` takes no kernel locks at all; it only calls the already-locked `xvstat(2)`). |
| 3 | Process telemetry is initialized at allocation and handled correctly on fork/exec/exit. | **Directly implemented and tested**, for both Phase 1 and Phase 2 telemetry. `allocproc()`/`freeproc()` zero `trace_mask` *and* the three Phase 2 counters. `kfork()` copies `trace_mask` (inherited) but deliberately leaves the Phase 2 counters at their zeroed value (not inherited -- see `docs/accounting.md` "fork/exec/exit semantics"); `kexec()` is untouched, so both survive exec by construction. Tested by `test_trace_fork_inheritance.py`/`test_trace_toggle.py` (Phase 1) and `test_fork_fresh_counters.py`/`test_exec_preserves_counters.py` (Phase 2). |
| 4 | The scheduler always eventually selects eligible work per documented policy assumptions. | Not modified through Phase 3 (Phase 4). Indirectly exercised: `test_trace_fork_inheritance.py`'s `forktest` run and every Phase 2/3 test's shell-driven fork/exec/wait cycles all need the baseline round-robin scheduler to keep making progress, and they do. |
| 5 | Zombie/free process slots never remain visible as active telemetry. | **Directly implemented and tested**, including a fix during Phase 3 review. `freeproc()` resets `trace_mask` and the three Phase 2 counters to 0 when a slot returns to `UNUSED` (`test_trace_isolation.py`, `test_slot_reuse_reset.py`). `xvtop`'s `collect()` (`user/xvtop.c`) filters both `XV_UNUSED` **and** `XV_ZOMBIE` slots out of its listing -- `ZOMBIE` is a real, non-`UNUSED` procstate and needs its own explicit check, which an earlier version of `collect()` was missing; `tests/xvtop/test_xvtop_zombie_filtered.py` is a dedicated regression test for exactly this, using a deterministically-constructed zombie (`user/xvtopzombie.c`). |
| 6 | VM changes preserve user/kernel isolation and mapping permissions. | Not applicable: Phases 1-3 make no VM changes (deferred to Phase 5, ADR-0005). |
| 7 | Syscall interfaces validate user pointers/arguments. | `trace(int mask)` (Phase 1) takes only a plain `int`, no pointer to validate, by design. `xvstat(int idx, struct xv_pstat *addr)` (Phase 2) has both: `idx` is bounds-checked by `procstat()` (`kernel/proc.c`), and `addr` is validated by `copyout()` itself, the same pattern `sys_fstat()` already uses. `tests/accounting/test_xvstat_bounds.py` drives all three rejection paths (`idx=-1`, `idx=NPROC`, a bad user pointer) plus the ordinary-success case, and confirms the syscall keeps working normally afterwards. |
| 8 | Observability code must not destabilize the kernel it observes. | **Directly tested**, Phase 1 through Phase 3. `test_no_regression.py` proves an untraced session behaves exactly like a stock boot. `test_trace_fork_inheritance.py` proves heavy tracing + proc-table-filling load doesn't destabilize anything. For Phase 2/3: `usertests` (upstream's own regression suite) reports `ALL TESTS PASSED` with accounting counters and their locking live on every process; `xvtop` and its six accounting test programs never crash or hang the kernel across `tests/accounting/` and `tests/xvtop/`, including the deliberately-constructed zombie/bad-pointer/out-of-range edge cases. |

## Known, deliberate limitations

`SYS_exit` can never produce a trace line: `syscall()` in
`kernel/syscall.c` prints the trace line *after* calling the
dispatched syscall handler, but `sys_exit()` -> `kexit()` never
returns to that call site -- it jumps directly into `sched()` and the
process is gone. This is not a bug to fix; it is a direct structural
consequence of how xv6 implements process exit, documented here so it
isn't mistaken for an oversight, and it does not violate invariant #8
(nothing unsafe is attempted -- the print statement is simply never
reached).

Phase 2's `syscall_count` handles this same structural fact the
opposite way, deliberately: it is incremented *before* dispatch
specifically so `SYS_exit` *is* counted (see `docs/accounting.md`
"Known limitation"). The consequence is that `exit` is always the
last syscall counted for a process -- there is no way for a later
event to increment the counter again, since the process is gone --
which is expected, not a bug.
