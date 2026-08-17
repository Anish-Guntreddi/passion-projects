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
