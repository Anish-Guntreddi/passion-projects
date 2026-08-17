# Core invariants

The eight invariants from spec §1.5, each guarded by tests/review, and
their status as of Phase 5. This document is updated every phase; a
stale entry here is a bug in the process, not just in the docs.

| # | Invariant | Status after Phase 5 |
|---|---|---|
| 1 | Kernel accounting never corrupts process lifecycle state. | **Directly implemented and tested**, through Phase 5. Phase 2 added `runticks`/`waitticks`/`syscall_count`; Phase 4 added `tickets`/`sched_selections`; Phase 5 added `pagefaults`/`pagefaults_failed` -- seven telemetry fields total on every `struct proc`. Upstream's own `usertests` reports `ALL TESTS PASSED` with all seven live and being written on every process throughout a ~199s run (`docs/scheduler.md`, `docs/vm-extension.md`). `tests/accounting/test_slot_reuse_reset.py`, `tests/scheduler/test_tickets_lifecycle.py`, and the Phase 5 fault-path tests (`tests/vm/`) directly prove none of this corrupts allocation/fork/exec/exit/free. |
| 2 | Locks are never acquired in inconsistent order. | **Directly addressed**, including one new lock. Phase 4 adds exactly one new lock, `sched_lock` (protects `sched_policy`/the lottery PRNG state) -- deliberately kept a leaf lock, never held while any `p->lock` is held and never acquiring one, so it cannot create a cycle with the existing `wait_lock -> p->lock` chain (ADR-0010). Phase 5 adds *no* new lock, and ADR-0012 documents a lock-order hazard that was found and specifically avoided: taking `p->lock` inside `vmfault()` would have let `kernel/proc.c`'s `kwait()` acquire two different processes' `p->lock` at once (the reaped child's, then -- via `copyout()`'s lazy fault on the parent's own buffer -- the waiting parent's), a lock-order edge that does not exist anywhere else in this codebase. |
| 3 | Process telemetry is initialized at allocation and handled correctly on fork/exec/exit. | **Directly implemented and tested**, for all four phases' telemetry. `allocproc()`/`freeproc()` zero every field (Phase 1's `trace_mask`, Phase 2's three counters, Phase 4's `tickets`/`sched_selections`, Phase 5's `pagefaults`/`pagefaults_failed`). `kfork()`'s per-field choice is now a three-way split, each with a stated reason (see `docs/scheduler.md`/`docs/accounting.md`): **inherited** -- `trace_mask` (Phase 1), `tickets` (Phase 4); **reset to 0, not inherited** -- the Phase 2 counters, `sched_selections` (Phase 4), `pagefaults`/`pagefaults_failed` (Phase 5). `kexec()` touches none of them -- all are per-pid-lifetime totals. Tested by `test_trace_fork_inheritance.py`/`test_fork_fresh_counters.py` (Phases 1/2), `test_tickets_lifecycle.py` (Phase 4), `test_fork_lazy_region.py`/`test_exec_discards_lazy_region.py` (Phase 5). |
| 4 | The scheduler always eventually selects eligible work per documented policy assumptions. | **Directly implemented and tested for the first time in Phase 4.** `schedule_lottery()`'s weighted draw can miss (a race between its two table passes) or have nothing to draw from (every `RUNNABLE` process has 0 tickets) -- both cases fall back to `pick_first_runnable()` (first-`RUNNABLE`-in-table-order), bounded by `SCHED_LOTTERY_MAX_ATTEMPTS` retries with that deterministic fallback as the unconditional last resort (ADR-0010). `tests/scheduler/test_lottery_zero_ticket_floor.py` is the direct proof: four 0-ticket processes alongside three 30-ticket ones all complete and are reaped, none hang. **Scope correction (review followup):** this invariant is about the *scheduler* never stalling on `RUNNABLE` work, which holds unconditionally -- it is *not* a claim that a 0-ticket process is individually guaranteed periodic CPU time under sustained competition from a persistently-`RUNNABLE` nonzero-ticket process (the fallback only fires when the total `RUNNABLE` ticket count is exactly 0). See `docs/scheduler.md`'s "Zero-ticket floor" section and ADR-0010 decision 5 for the corrected, precise statement of what this feature does and does not guarantee. Also see [ADR-0013](decisions/0013-ticket-count-bound.md): `settickets(2)` now rejects values above `SCHED_MAX_TICKETS` to keep the weighted draw's uint32 PRNG modulo from truncating a large `RUNNABLE` ticket total, which would otherwise make some processes structurally unreachable by any draw -- a distinct correctness gap from the zero-ticket case, also found and fixed during the same review followup. |
| 5 | Zombie/free process slots never remain visible as active telemetry. | **Directly implemented and tested**, through Phase 5. `freeproc()` resets every telemetry field (all seven, across four phases) to 0 when a slot returns to `UNUSED`. `xvtop`'s zombie/unused filtering (Phase 3) is unaffected by the Phase 4/5 field additions -- `xvtop` simply doesn't display the new columns, it was never changed. Slot-reuse-after-nonstandard-value is now tested per phase where it applies: `test_slot_reuse_reset.py` (Phase 2 counters), `test_tickets_lifecycle.py` (Phase 4: a slot that previously held `settickets(77)` must show the default 10 on reuse, not 77). |
| 6 | VM changes preserve user/kernel isolation and mapping permissions. | **Directly tested for the first time in Phase 5** -- this already-shipping upstream fault path (ADR-0002/ADR-0011: the lazy-allocation mechanism itself predates this project) had zero dedicated tests before Phase 5. `test_oob_access_killed.py` proves an out-of-bounds fault is recognized and kills the offending process cleanly, not silently allowed through; `test_memory_exhaustion_recovery.py` proves a killed process's mapped pages are fully reclaimed on reap (no cross-process leak); `test_fork_lazy_region.py` proves a child's own fault-in is isolated to the child's own page table, not shared with the parent's. **Review followup:** the "mapping permissions" half of this invariant had a real gap -- `ismapped()` only checked `PTE_V`, so a genuine permission violation (e.g. a process writing to its own read-only text segment) fell into `vmfault()`'s "already mapped" branch, was neither counted into `pagefaults_failed` nor distinguished from an out-of-bounds/out-of-memory failure, and got a factually wrong kill diagnostic. `vm_permission_violation()` (`kernel/vm.c`) now checks the actual permission bit the faulting access needs against what's mapped; `kernel/trap.c` prints a distinct, accurate message for this case; and `test_permission_fault_killed.py`/`user/vmpermtest.c` directly exercise it (a store to the process's own text segment). `test_pagefault_failed_counted.py`/`user/vmpfailcount.c` additionally closes a related test-coverage gap: every pre-existing failure-path test only *inferred* `pagefaults_failed` incremented from a killed process's exit status; this one reads the counter directly, before and after, in a process that survives (via a deliberately out-of-bounds `xvstat(2)` output pointer, which fails the syscall without killing the caller). |
| 7 | Syscall interfaces validate user pointers/arguments. | Two more int-only syscalls added in Phase 4 (`settickets(int)`, `schedpolicy(int)`, same "no user pointer to mis-validate" shape as Phase 1's `trace()`), each with its own real argument-value validation now directly tested (`test_settickets_validation.py`, `test_schedpolicy_validation.py`) -- negative tickets and out-of-range policy values are rejected without side effects. Phase 5 adds **no new syscall** at all (page-fault telemetry rides the existing `xvstat(2)` interface, ADR-0012), so no new argument-validation surface. |
| 8 | Observability code must not destabilize the kernel it observes. | **Directly tested**, Phase 1 through Phase 5. `test_no_regression.py` (Phase 1), `usertests` `ALL TESTS PASSED` (Phases 2-5, ~199s this build), and every Phase 4/5 test that deliberately triggers a failure path (`vmoobtest`'s out-of-bounds kill, `vmexhausttest`'s real memory exhaustion, `tixvalidate`'s rejected `schedpolicy(99)`) explicitly follows up with a plain shell command and asserts it still works. Phase 5 additionally hardened, rather than just observed: `kernel/trap.c`'s diagnostic split gives a recognized-but-failed page fault its own clear message instead of the generic "unexpected scause" line -- confirmed live in `usertests`' own `lazytests` OOM case, which still reports `OK`. |

## Known, deliberate limitations

`SYS_exit` can never produce a Phase 1 trace line, for the structural
reason `kexit()` never returns to `syscall()`'s post-dispatch print
site (documented since Phase 1). Phase 2's `syscall_count` handles the
same structural fact the opposite way (counts *before* dispatch, so
`exit` *is* counted, necessarily as the last syscall for that
process). Both are pre-existing, unchanged by Phase 4/5.

**Phase 4:** `sched_policy` is read in `scheduler()`'s hot loop without
`sched_lock` (a plain, naturally-aligned `int`) -- the same
"best-effort, not safety-critical" reasoning ADR-0009 already applies
to the global `ticks`. A policy switch becoming visible to one hart an
iteration later than another is an accepted, harmless staleness, not a
correctness gap: invariant #8 asks that a policy *change* not
destabilize the kernel, not that it propagate instantaneously.

**Phase 5:** `pagefaults`/`pagefaults_failed` are, like `sz`/`name`
before them (ADR-0009), read cross-process by `xvstat(2)` without a
*hard* torn-read guarantee on the write side -- deliberately, per
ADR-0012's lock-order analysis, not an oversight. A `vmfault()` call
triggered from inside `copyout()`/`copyin()` (as opposed to a genuine
user-space page fault via `usertrap()`) is real and telemetered
identically; this project does not currently distinguish "the
process's own instruction faulted" from "a syscall handler's copy
faulted on the process's behalf" in the counters themselves.

**Phase 4, review followup:** the "zero-ticket floor" is real but
narrower than its original wording claimed -- it guarantees the
*scheduler* never stalls on `RUNNABLE` work, not that any individual
0-ticket process gets periodic CPU time under sustained competition
from a persistently-`RUNNABLE` nonzero-ticket process. See row #4
above and `docs/scheduler.md` for the corrected statement.

**Phase 5, review followup:** `vm_permission_violation()`
(`kernel/vm.c`) re-derives its classification with a second `walk()`
call, separate from the one `vmfault()` itself already did, rather
than threading a result through. This is deliberately simple rather
than plumbing new state through `vmfault()`'s return value, and is
safe specifically because only the one hart currently running a given
process can be mutating that process's own page table at fault time
(the same reasoning ADR-0012 already relies on for the counters
themselves being lock-free) -- there is no new race introduced by
checking twice.
