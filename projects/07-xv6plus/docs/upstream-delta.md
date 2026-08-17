# Upstream delta

This document is the load-bearing honesty record required by spec
§1.2: it distinguishes **(A)** upstream xv6 code, **(B)** completed
educational lab modifications, and **(C)** original xv6-plus
extensions. See ADR-0008 for why this document (plus inline
`// xv6-plus:` comments) stands in for a branch-per-category structure
in this monorepo.

**Pinned base:** `mit-pdos/xv6-riscv` @ tag `xv6-riscv-rev5`, commit
`7d7adbb1b0acbd67c9766a20d0f9900fef2789fa` (ADR-0001).

## (A) Upstream xv6 code -- unmodified

Everything under `kernel/`, `user/`, `mkfs/` not listed under (C)
below, plus `Makefile` (except the one line noted under (C)), `README`
(the file baked into the xv6 filesystem image, at the project root, no
extension), `LICENSE.upstream-xv6`, `.gdbinit.tmpl-riscv`,
`.editorconfig`, `.dir-locals.el`. Copied verbatim from the pinned
commit; not hand-retyped, not "cleaned up," not reformatted.

Note this pinned revision is itself already meaningfully different
from the "classic" xv6-riscv many public writeups describe -- e.g.
`sleep()` is named `pause()`, `fork`/`exit`/`wait`/`exec` are
implemented by `kfork`/`kexit`/`kwait`/`kexec` (with `sys_*` wrappers
calling them), and `sbrk()` has an eager/lazy mode split
(`kernel/vm.h`: `SBRK_EAGER`/`SBRK_LAZY`). All kernel/user
modifications below were written against *this* pinned tree, read
directly beforehand (per handoff rule 1), not against a
half-remembered older xv6.

## (B) Completed educational lab modifications

**None.** See ADR-0002: this implementer had no information about
which, if any, standard course labs had been separately completed, so
the conservative and honest default is zero incorporated labs. The
pinned tree itself contains no lab-specific scaffolding (verified by
inspection after vendoring).

## (C) Original xv6-plus extensions

### Phase 0 (reproducible base)

No code changes. Deliverables: the pin itself (ADR-0001), this
document, `docs/decisions/`, `docs/toolchain.md`, and the test
harness (`scripts/qemu_session.py`, `scripts/run_tests.py`,
`scripts/run-tests.sh`) that proves the pin builds and boots.

Also rewritten (not vendored verbatim) for monorepo safety:
`.gitignore` -- see the comment at the top of that file and ADR-0008
for why upstream's own `.gitignore` isn't reused as-is (its bare
`mkfs` pattern would silently exclude the whole `mkfs/` directory,
`mkfs.c` included, from ever being committed in a fresh monorepo add).

### Phase 1 (syscall tracing foundation, FR1)

New files:

| File | What |
|---|---|
| `user/xtrace.c` | Userspace control tool: `xtrace mask cmd [args...]` enables tracing then `exec()`s into `cmd`. |
| `user/tracetest.c` | Direct enable/disable unit-test program for `trace()`, used by `tests/syscall/test_trace_toggle.py`. |
| `tests/syscall/*.py`, `tests/syscall/_helpers.py` | Phase 1 automated test suite (see `docs/tracing.md`). |
| `scripts/qemu_session.py`, `scripts/run_tests.py`, `scripts/run-tests.sh` | Test harness (also covers Phase 0's boot smoke test). |

Touched files (all changes marked inline with `// xv6-plus:`
comments):

| File | Change |
|---|---|
| `kernel/proc.h` | Added `int trace_mask;` to `struct proc`. |
| `kernel/proc.c` | `allocproc()`: zero `trace_mask` on allocation. `freeproc()`: zero `trace_mask` on free. `kfork()`: copy `trace_mask` from parent to child. |
| `kernel/syscall.h` | Added `#define SYS_trace 22`. |
| `kernel/syscall.c` | Added `sys_trace` to the dispatch table; added a `syscall_names[]` table; `syscall()` now prints `"pid: syscall NAME -> RET"` after a traced syscall returns, gated on `(p->trace_mask >> num) & 1`. |
| `kernel/sysproc.c` | Added `sys_trace()`: sets `myproc()->trace_mask = mask`. |
| `user/user.h` | Added `int trace(int);` prototype. |
| `user/usys.pl` | Added `entry("trace");` (generates the `trace()` syscall stub). |
| `Makefile` | Added `$U/_xtrace` and `$U/_tracetest` to `UPROGS`. |

**ABI change:** one new syscall number (22, `SYS_trace`); no existing
syscall numbers, signatures, or return-value conventions changed.

**Invariants at stake:** #3 (telemetry init/fork/exec/exit), #5
(zombie/free slots never show stale telemetry), #7 (the only argument
is a plain `int`, so there is no user pointer to mis-validate), #8
(tracing must not destabilize the kernel it observes). See
`docs/invariants.md` for the full mapping and `docs/tracing.md` for
the design writeup, safe-scope rationale, and a known limitation
(`SYS_exit` is never traced, because `kexit()` never returns to the
print site).

**Lock/lifetime:** no new lock introduced; `trace_mask` follows the
existing "private to the process" convention already used for `sz`,
`pagetable`, `ofile`, `cwd`, `name` in `kernel/proc.h` (ADR-0006).

**Test:** `tests/syscall/test_trace_basic.py`,
`test_trace_isolation.py`, `test_trace_fork_inheritance.py`,
`test_trace_toggle.py`, `test_no_regression.py`. Run via
`scripts/run-tests.sh`.

**Rollback/debug strategy:** every xv6-plus change in `kernel/` is
additive and separately identifiable via `// xv6-plus:` comments;
reverting Phase 1 means removing the `trace_mask` field, the three
`proc.c` touch points, the `SYS_trace` entry, and the two new user
programs -- upstream behavior is otherwise untouched, so a revert
cannot regress non-tracing functionality. For interactive debugging,
`make qemu-gdb` + `riscv64-unknown-elf-gdb -x .gdbinit` from a second
terminal works unmodified against this tree (see `docs/toolchain.md`).

### Phase 2 (process accounting, FR2/FR3)

New files:

| File | What |
|---|---|
| `kernel/pstat.h` | `struct xv_pstat`: the fixed-layout snapshot struct `xvstat(2)` copies out to userspace. |
| `user/acct.h` | Shared header for the accounting test programs and `xvtop`: `XV_*` procstate mirrors and `xv_find_self()`, a small `static inline` helper that scans the proc table via `xvstat()` for the caller's own entry. |
| `user/accttest.c` | Reports its own `struct xv_pstat` at three stages (start / after five `getpid()`s / after `pause(5)`) to exercise `syscall_count` and `waitticks`. |
| `user/acctforktest.c` | Proves a forked child's counters start at 0, not copied from the parent's already-accumulated totals. |
| `user/acctexectest.c`, `user/acctreport.c` | A two-program pair proving counters survive `exec()`: `acctexectest` runs a known number of syscalls then `exec()`s into `acctreport`, which reports the inherited count. |
| `user/xvstatbounds.c` | Exercises `xvstat(2)`'s argument validation: out-of-range index, bad user pointer, and the ordinary-success case. |
| `tests/accounting/*.py`, `tests/accounting/_acct_helpers.py` | Phase 2 automated test suite (see `docs/accounting.md`). Named `_acct_helpers`, not `_helpers`, specifically to avoid a `sys.modules` name collision with `tests/syscall/_helpers.py` in `scripts/run_tests.py`'s single-process test run. |
| `docs/accounting.md` | Phase 2 design writeup: kernel side, `xvstat(2)` interface, fork/exec/exit semantics, the `SYS_exit`-counts-itself limitation, locking summary, test coverage. |
| `docs/decisions/0009-accounting-counter-locking.md` | ADR: why every accounting-counter write takes `p->lock` explicitly, and why `tickslock` reuse for the `waitticks` write was considered and rejected (lock-order hazard against `clockintr()`'s `tickslock -> p->lock` chain). |

Touched files (all changes marked inline with `// xv6-plus:` comments):

| File | Change |
|---|---|
| `kernel/proc.h` | Added `uint64 runticks`, `uint64 waitticks`, `uint64 syscall_count` to `struct proc`. |
| `kernel/proc.c` | `allocproc()`: zero the three counters on allocation. `freeproc()`: zero them again on free. `kfork()`: deliberately does *not* copy them to the child (documented at the call site, contrasting with `trace_mask` just above it). `sleep()`: snapshots `ticks` unlocked before sleeping, adds the delta to `p->waitticks` (under `p->lock`) on wakeup. Added `procstat()`: fills a `struct xv_pstat` from `proc[idx]` under that slot's `p->lock`. |
| `kernel/sysproc.c` | Added `sys_xvstat()`: reads `idx`/`addr`, calls `procstat()`, `copyout()`s the result. |
| `kernel/syscall.c` | Added `sys_xvstat` to the dispatch table and `syscall_names[]`. `syscall()` now increments `p->syscall_count` (under `p->lock`) before dispatching every syscall. |
| `kernel/syscall.h` | Added `#define SYS_xvstat 23`. |
| `kernel/trap.c` | `clockintr()`: charges one `runtick` (under `p->lock`) to whichever process is running on that hart, every timer interrupt. |
| `kernel/defs.h` | Added the `struct xv_pstat` forward declaration and `procstat()` prototype. |
| `user/user.h` | Added `struct xv_pstat` forward declaration and `int xvstat(int, struct xv_pstat*);` prototype. |
| `user/usys.pl` | Added `entry("xvstat");` (generates the `xvstat()` syscall stub). |
| `Makefile` | Added `$U/_accttest`, `$U/_acctforktest`, `$U/_acctexectest`, `$U/_acctreport`, `$U/_xvstatbounds` to `UPROGS`. (`$U/_xvtop` is listed here too but belongs to Phase 3, below.) |
| `docs/invariants.md` | Updated all eight rows for Phase 2 status. |
| `README.md` | Added the FR2/FR3 "Original features" entry; updated the roadmap status table. |
| `docs/decisions/0003-stats-interface.md`, `docs/decisions/0006-telemetry-synchronization.md`, `docs/decisions/0007-counter-granularity.md` | Status lines updated from "deferred to Phase 2" to "implemented in Phase 2", now that the interface/locking/granularity decisions they recorded ahead of time have actually been built against. |

**ABI change:** one new syscall number (23, `SYS_xvstat`); no existing
syscall numbers, signatures, or return-value conventions changed.

**Invariants at stake:** #1 (accounting must never corrupt process
lifecycle state -- `usertests` still reports `ALL TESTS PASSED` with
these fields live on every process), #2 (no new lock is introduced;
every counter write reuses `p->lock`, see ADR-0009 for why `tickslock`
reuse was specifically rejected), #3 (counters initialized at
allocation, correctly handled on fork -- reset, not copied -- and exec
-- preserved), #5 (zombie/free slots reset on free, same as
`trace_mask`), #7 (`xvstat(2)` validates both its `idx` and its user
pointer). See `docs/invariants.md` for the full mapping and
`docs/accounting.md` for the design writeup.

**Lock/lifetime:** every accounting-counter write takes the target
process's own `p->lock`, per ADR-0006 case (b) and ADR-0009's full
reasoning -- a new pattern relative to Phase 1's lock-free
`trace_mask`, made necessary because these counters (unlike
`trace_mask`) are read cross-process by `xvstat(2)`.

**Test:** `tests/accounting/test_slot_reuse_reset.py`,
`test_fork_fresh_counters.py`, `test_syscall_count_monotonic.py`,
`test_waitticks_from_pause.py`, `test_exec_preserves_counters.py`,
`test_xvstat_bounds.py`. Run via `scripts/run-tests.sh`. Also verified
manually against upstream's own `usertests` regression suite (`ALL
TESTS PASSED`, no regression from the new `proc.h` fields/locking).

**Rollback/debug strategy:** additive and separately identifiable via
`// xv6-plus:` comments, same discipline as Phase 1; reverting Phase 2
means removing the three `struct proc` counters, their five
lifecycle/accounting touch points in `proc.c`/`syscall.c`/`trap.c`,
the `SYS_xvstat` entry and `sys_xvstat()`, `kernel/pstat.h`, and the
five new Phase-2 user programs -- Phase 1 tracing and all upstream
behavior are otherwise untouched.

### Phase 3 (`xvtop`, FR4)

New files:

| File | What |
|---|---|
| `user/xvtop.c` | The `xvtop` userspace monitor: polls `xvstat(2)` across `idx = 0..NPROC-1`, filters `UNUSED`/`ZOMBIE` slots, sorts by `runticks` descending, prints one row per surviving process plus a summary line. |
| `user/xvtopzombie.c` | Deterministic support program for the zombie-filtering regression test: forks a child that exits immediately and is never `wait()`ed on, polls for it to reach `XV_ZOMBIE`, then `exec()`s into `xvtop` so the zombie is guaranteed present for `xvtop`'s first refresh. |
| `tests/xvtop/test_xvtop_basic.py`, `test_xvtop_zombie_filtered.py` | Phase 3 automated test suite (see `docs/xvtop.md`). |
| `docs/xvtop.md` | Phase 3 design writeup: behavior, example transcript, zombie-filtering rationale, known limitation (no single atomic whole-table snapshot), test coverage. |

Touched files:

| File | Change |
|---|---|
| `Makefile` | Added `$U/_xvtop` and `$U/_xvtopzombie` to `UPROGS`. |
| `README.md` | Added the FR4 "Original features" entry; updated the roadmap status table. |

**ABI change:** none -- `xvtop` is a pure consumer of the Phase 2
`xvstat(2)` interface, no new syscall.

**Invariants at stake:** #5 (zombie/free process slots must never
remain visible as active telemetry -- directly what
`test_xvtop_zombie_filtered.py` exists to prove: `collect()` filters
both `XV_UNUSED` and `XV_ZOMBIE`, not just the former), #8
(observability must not destabilize the kernel it observes -- `xvtop`
only ever calls the already-validated `xvstat(2)`, no new kernel
surface). See `docs/xvtop.md` for the full design writeup.

**Lock/lifetime:** none of `xvtop`'s own code touches a kernel lock;
it is entirely userspace, reading a per-slot-consistent snapshot from
each `xvstat(2)` call (ADR-0009's guarantee), composited client-side
into one refresh (documented as a known limitation in `docs/xvtop.md`:
no single atomic whole-table snapshot).

**Test:** `tests/xvtop/test_xvtop_basic.py`,
`test_xvtop_zombie_filtered.py`. Run via `scripts/run-tests.sh`.

**Rollback/debug strategy:** additive; reverting Phase 3 means
removing `user/xvtop.c`, `user/xvtopzombie.c`, and their two `UPROGS`
entries -- Phase 1/2 and all upstream behavior are otherwise
untouched.

### Phase 4 (scheduler experiment, FR5)

New files:

| File | What |
|---|---|
| `kernel/sched.h` | `SCHED_RR`/`SCHED_LOTTERY` policy constants, `SCHED_DEFAULT_TICKETS` -- shared kernel/user header, same convention as `kernel/vm.h`'s `SBRK_EAGER`/`SBRK_LAZY`. |
| `user/schedbench.c` | Benchmark driver: sets a policy, forks one child per ticket count, each child spins a fixed wall-clock window, self-reports via a private pipe (avoids cross-process console interleaving -- see `docs/scheduler.md`). |
| `user/tixtest.c` | Lifecycle test program: tickets inherited across `fork()`, reset to default on slot reuse. |
| `user/tixvalidate.c` | `settickets()`/`schedpolicy()` argument-validation test program. |
| `tests/scheduler/*.py`, `tests/scheduler/_sched_helpers.py` | Phase 4 automated test suite (see `docs/scheduler.md`). |
| `docs/scheduler.md` | Phase 4 design writeup: policy switch, PRNG, lifecycle, zero-ticket floor, calibration, captured benchmark data. |
| `docs/decisions/0010-lottery-scheduler-design.md` | ADR: PRNG choice, two-pass-draw-with-bounded-retry design, tickets/selections lifecycle, zero-ticket floor -- the implementation questions ADR-0004 deferred. |

Touched files (all changes marked inline with `// xv6-plus:`
comments):

| File | Change |
|---|---|
| `kernel/proc.h` | Added `int tickets`, `uint64 sched_selections` to `struct proc`. |
| `kernel/proc.c` | New globals `sched_policy`/`sched_lock`/`rng_state`. `allocproc()`/`freeproc()`: init/reset both fields. `kfork()`: `tickets` inherited (like `trace_mask`), `sched_selections` not (like the Phase 2 counters). `scheduler()`'s loop split into `schedule_roundrobin()` (upstream logic, unchanged, refactored) and new `schedule_lottery()`/`lottery_rand()`/`runnable_ticket_total()`/`draw_and_run()`/`pick_first_runnable()`/`run_selected()`. Added `sched_settickets()`/`sched_setpolicy()`. `procstat()` extended to fill `tickets`/`selections`. |
| `kernel/sysproc.c` | Added `sys_settickets()`, `sys_schedpolicy()`. |
| `kernel/syscall.h` | Added `#define SYS_settickets 24`, `#define SYS_schedpolicy 25`. |
| `kernel/syscall.c` | Added both to the dispatch table and `syscall_names[]`. |
| `kernel/defs.h` | Added `sched_settickets()`/`sched_setpolicy()` prototypes. |
| `kernel/pstat.h` | `struct xv_pstat` gains `tickets`/`selections`. |
| `user/user.h` | Added `int settickets(int)`, `int schedpolicy(int)` prototypes. |
| `user/usys.pl` | Added `entry("settickets")`, `entry("schedpolicy")`. |
| `user/acct.h` | Comment broadened: `xv_find_self()` is now reused by Phase 4 (and Phase 5) programs too, no logic change. |
| `Makefile` | Added `$U/_schedbench`, `$U/_tixtest`, `$U/_tixvalidate` to `UPROGS`. |
| `docs/invariants.md` | Updated all eight rows through Phase 4/5 status. |
| `docs/decisions/0004-scheduler-policy.md` | Status updated from "implementation deferred" to "implemented," pointing at ADR-0010. |
| `README.md` | Added the FR5 "Original features" entry; updated the roadmap status table. |
| `tests/scheduler/README.md` | Removed (placeholder retired now that the phase is implemented -- matches the convention Phases 1-3's test directories already follow: no placeholder README once a directory has real tests). |

**ABI change:** two new syscall numbers (24 `SYS_settickets`, 25
`SYS_schedpolicy`); no existing syscall numbers, signatures, or
return-value conventions changed. `struct xv_pstat` (Phase 2, ADR-0003)
gains two fields, appended at the end -- purely additive, no existing
field's offset or meaning changes.

**Invariants at stake:** #2 (one new lock, `sched_lock`, deliberately
scoped as a leaf lock -- see ADR-0010), #3 (tickets/selections
init/fork/exec/exit semantics), #4 (directly tested for the first
time: the zero-ticket-floor fallback), #5 (slot reuse resets both new
fields), #7 (both new syscalls' argument validation is tested), #8
(regression-checked after every deliberately-adversarial test case).
See `docs/invariants.md` for the full mapping and `docs/scheduler.md`
for the design writeup.

**Lock/lifetime:** one new lock, `sched_lock`, protecting `sched_policy`
and the lottery PRNG state -- deliberately never held while any
`p->lock` is held, and nothing reachable while holding it acquires a
`p->lock`, so it introduces no new edge in the existing
`wait_lock -> p->lock` lock-order graph. `tickets` takes `p->lock` on
write (ADR-0006 case (b): read cross-process by the scheduler as a
real input, not just monitored); `sched_selections` is written only by
`scheduler()` itself while already holding the target process's own
`p->lock` (no separate acquire needed, same pattern as `runticks`).

**Test:** `tests/scheduler/test_tickets_lifecycle.py`,
`test_settickets_validation.py`, `test_schedpolicy_validation.py`,
`test_sched_baseline_fairness.py`, `test_sched_lottery_fairness.py`,
`test_lottery_zero_ticket_floor.py`. Run via `scripts/run-tests.sh`.
Also re-verified against upstream's own `usertests` regression suite
(`ALL TESTS PASSED`, ~223.5s, this build) with all Phase 4 fields/locks
live throughout (baseline `SCHED_RR` policy the whole run, since
`usertests` never calls `schedpolicy()`).

**Rollback/debug strategy:** additive and separately identifiable via
`// xv6-plus:` comments, same discipline as every earlier phase.
Reverting Phase 4 means removing the two `struct proc` fields, the
`sched_lock`/`sched_policy`/`rng_state` globals, the
`schedule_lottery()` call graph (leaving `schedule_roundrobin()`, which
is upstream's own loop, as the sole scheduler), the two new syscalls,
and the three new Phase 4 user programs -- upstream's round-robin
behavior and every earlier phase are otherwise untouched (indeed
`schedule_roundrobin()` *is* upstream's own loop, unmodified in
control flow).

### Phase 5 (VM extension: page-fault telemetry, FR6)

New files:

| File | What |
|---|---|
| `user/vmfaulttest.c` | Basic telemetry correctness: `sbrklazy()` alone counts nothing; touching N pages counts exactly N; re-touching doesn't double-count. |
| `user/vmoobtest.c` | Out-of-bounds access: a child dereferencing far beyond its own `sz` is killed cleanly; the parent/session stays healthy. |
| `user/vmforktest.c` | Fork isolation: a child's own fault-in of an inherited-but-unmapped lazy region is independent of the parent's counters. |
| `user/vmexectest.c` | Exec safety: a pending, never-touched lazy region doesn't crash or leak across `exec()`. |
| `user/vmexhausttest.c` | Real memory exhaustion + reclaim-on-exit proof. |
| `tests/vm/*.py` | Phase 5 automated test suite (see `docs/vm-extension.md`). |
| `docs/vm-extension.md` | Phase 5 design writeup: honesty-layer split (what's upstream vs. original in this feature), locking, lifecycle, memory-exhaustion test configuration, captured real runs. |
| `docs/decisions/0011-vm-extension-choice.md` | ADR: finalizes D5 (page-fault telemetry), superseding ADR-0005's deferral. |
| `docs/decisions/0012-pagefault-telemetry-locking.md` | ADR: why the new counters are deliberately lock-free (a real lock-order hazard against `kwait()`'s `copyout()`, not a style choice). |

Touched files (all changes marked inline with `// xv6-plus:`
comments):

| File | Change |
|---|---|
| `kernel/proc.h` | Added `uint64 pagefaults`, `uint64 pagefaults_failed` to `struct proc`. |
| `kernel/proc.c` | `allocproc()`/`freeproc()`: init/reset both fields. `kfork()`: neither inherited (comment only -- both are already 0 from `allocproc()`, same as the Phase 2 counters). `procstat()` extended to fill both. |
| `kernel/vm.c` | `vmfault()`: increments `p->pagefaults`/`p->pagefaults_failed` at each of its three outcomes. No change to the actual allocation/mapping logic (that part is upstream, category (A)). |
| `kernel/trap.c` | `usertrap()`'s combined `(scause==13\|\|15) && vmfault(...)!=0` condition split into its own branch with a specific diagnostic on failure -- same continue-on-success/kill-on-failure decision, clearer message only. |
| `kernel/pstat.h` | `struct xv_pstat` gains `pagefaults`/`pagefaults_failed`. |
| `docs/invariants.md` | Updated all eight rows; invariant #6 gets real test coverage for the first time. |
| `docs/decisions/0005-vm-extension.md` | Status updated to "Superseded by ADR-0011." |
| `README.md` | Added the FR6 "Original features" entry; updated the roadmap status table. |
| `tests/vm/README.md` | Removed (same convention as `tests/scheduler/README.md` above). |

**ABI change:** **none** -- no new syscall. `struct xv_pstat` gains
two fields, appended at the end (purely additive, same as Phase 4's
addition to the same struct).

**Invariants at stake:** #1 (accounting never corrupts process
lifecycle state -- `usertests` `ALL TESTS PASSED` with the new fields
live), #2 (a lock-order hazard was found and specifically avoided, not
just "no new lock was added" -- see ADR-0012), #3 (neither field
inherited across `fork()`, matching the Phase 2 counter pattern), #5
(reset on free), #6 (directly tested for the first time in this
project: out-of-bounds kill, exhaustion+reclaim, fork isolation), #8
(every deliberately-adversarial Phase 5 test case is followed by a
regression check, and the `kernel/trap.c` diagnostic split is itself a
small hardening of observability without changing any control-flow
decision). See `docs/invariants.md` and `docs/vm-extension.md`.

**Lock/lifetime:** no new lock. The two counters are deliberately
lock-free (ADR-0012): mutated only by `vmfault()` acting on
`myproc()`, so there is no concurrent-writer race; read cross-process
by `xvstat(2)` with the same accepted best-effort staleness ADR-0009
already established for `sz`/`name` in the same struct -- not a new
precedent, an extension of an existing one, chosen specifically
because taking `p->lock` inside `vmfault()` would have introduced a
lock-order edge `kwait()`'s `copyout()` doesn't currently have.

**Test:** `tests/vm/test_pagefault_counting.py`,
`test_oob_access_killed.py`, `test_fork_lazy_region.py`,
`test_exec_discards_lazy_region.py`,
`test_memory_exhaustion_recovery.py`. Run via `scripts/run-tests.sh`.
Also re-verified against upstream's own `usertests` regression suite
(`ALL TESTS PASSED`, ~223.5s, this build) -- notably including
`usertests`' own `lazytests` group, which exhausts memory via this
exact `vmfault()` path and now prints this project's new, more
specific diagnostic message instead of upstream's generic one, while
still reporting `OK`.

**Rollback/debug strategy:** additive and separately identifiable via
`// xv6-plus:` comments. Reverting Phase 5 means removing the two
`struct proc` fields, their three touch points in
`proc.c`/`vm.c`/`trap.c` (`trap.c`'s revert restores upstream's single
combined condition exactly), and the five new Phase 5 user programs --
the underlying lazy-allocation mechanism itself is upstream and
untouched either way, so a revert cannot regress `sbrk()`/lazy-fault
behavior, only this project's telemetry and tests of it.

### Phase 4/5 review-fix followup (correctness gaps found by external review)

New files:

| File | What |
|---|---|
| `user/vmpermtest.c` | Permission-violation test: a child writing to its own read-only text segment is killed with an accurate diagnostic, not the generic OOB/OOM one. |
| `user/vmpfailcount.c` | Direct `pagefaults_failed` counter verification (before/after, in a process that survives the failure it triggers -- via a deliberately out-of-bounds `xvstat(2)` output pointer). |
| `tests/vm/test_permission_fault_killed.py` | Drives `vmpermtest.c`. |
| `tests/vm/test_pagefault_failed_counted.py` | Drives `vmpfailcount.c`. |
| `docs/decisions/0013-ticket-count-bound.md` | ADR: `SCHED_MAX_TICKETS`, closing a PRNG-modulo-truncation correctness gap in the lottery draw. |

Touched files (all changes marked inline with `// xv6-plus:` comments):

| File | Change |
|---|---|
| `kernel/sched.h` | Added `SCHED_MAX_TICKETS` (100000). |
| `kernel/proc.c` | `sched_settickets()` now also rejects `n > SCHED_MAX_TICKETS`. |
| `kernel/vm.c` | New `vm_permission_violation(pagetable, va, read)` helper; `vmfault()`'s "already mapped" branch now counts `pagefaults_failed` only for a genuine permission violation (checked via the new helper), not for the benign concurrent-fault-race case. **A real regression was introduced and then caught by a full `usertests` re-run** (this project's own required verification step for any `vmfault()`/`trap.c` change): the first version of `vm_permission_violation()` called `walk()` directly without the `va >= MAXVA` guard `walkaddr()` (this same file) already carries, so `usertests`' `MAXVAplus` case -- which faults on an address `>= MAXVA` without growing `p->sz`, reaching `kernel/trap.c`'s new call to this helper with that raw address -- panicked the kernel (`walk()` panics unconditionally on `va >= MAXVA`) instead of cleanly killing the process. Fixed before this followup was considered done; see `docs/vm-extension.md`'s "Test coverage" section for the full account. |
| `kernel/trap.c` | The page-fault failure branch now calls `vm_permission_violation()` to print a distinct, accurate diagnostic for a permission violation vs. an out-of-bounds/out-of-memory failure. |
| `kernel/defs.h` | Declares `vm_permission_violation()`. |
| `kernel/proc.h` | `pagefaults_failed` field comment updated to mention the permission-violation case. |
| `user/schedbench.c` | `MAX_CHILDREN` corrected from a hardcoded, wrong `30` (`MAXARG - 2`) to `NOFILE - 4` (12), reflecting this program's real per-process fd budget; verified directly against a real build (see `docs/scheduler.md`). |
| `user/tixvalidate.c` | Added `settickets(SCHED_MAX_TICKETS +/- boundary)` cases. |
| `tests/scheduler/test_settickets_validation.py` | Asserts the new ticket-count-bound cases. |
| `docs/scheduler.md` | Corrected the "Captured real run" transcript block to match `benchmarks/raw/scheduler/*.txt` (an earlier hand-typed version did not); corrected the "monotonic-by-ticket-tier" narrative claim to "monotonic-by-tier-*average*" (individual runs are noisier); rewrote "Zero-ticket floor" to state the precise, narrower guarantee; corrected `MAX_CHILDREN`'s documented capacity. |
| `README.md` | Same transcript correction (FR5 section); softened the zero-ticket-floor claim; updated FR6 section and the "Verify it yourself" transcript (25/25 -> 27/27, two new tests). |
| `docs/decisions/0010-lottery-scheduler-design.md` | Decision 5 rewritten to state the zero-ticket floor's precise scope. |
| `docs/invariants.md` | Invariant #4 and #6 rows, and the "Known, deliberate limitations" section, updated with the corrections above. |
| `docs/vm-extension.md` | New review-followup subsection, coverage-table row split (invalid address vs. permission), new captured transcripts, diagnostic-message description updated. |
| `benchmarks/methodology.md` | Unaffected -- independently re-verified during this same review and found already correct; the bug was only in the two docs above, which now match it. |

**Why these are correctness/documentation fixes, not new features:**
each item above was found by an external review of the Phase 4/5 work
already described earlier in this file; none of them changes what
Phase 4/5 were trying to build, only whether the code and docs
actually deliver what they claim.
