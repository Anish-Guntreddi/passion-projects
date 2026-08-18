# xv6-plus

Extended teaching kernel and systems observatory, built on MIT's xv6
RISC-V teaching kernel. Portfolio project 07 of 09 (Track D: OS,
independent of the other projects). Full spec: `07-xv6plus-spec.md`
(portfolio repo root).

**Status: Phases 0-7 of 9 complete.** This is *not* a finished
project -- see "Roadmap status" below. Everything claimed here is
backed by a real, reproducible build and test run in this repo; see
"Verify it yourself."

## Honesty statement (spec §1.2)

This repo distinguishes three categories of code, tracked in
[`docs/upstream-delta.md`](docs/upstream-delta.md):

- **(A) Upstream xv6 code** -- `mit-pdos/xv6-riscv` @ tag
  `xv6-riscv-rev5`, vendored verbatim. See
  [`docs/decisions/0001-upstream-revision-pin.md`](docs/decisions/0001-upstream-revision-pin.md).
- **(B) Completed educational lab modifications** -- none. See
  [`docs/decisions/0002-course-lab-incorporation.md`](docs/decisions/0002-course-lab-incorporation.md).
- **(C) Original xv6-plus extensions** -- everything listed under
  "Original features" below. Every original change is also marked
  inline in the source with a `// xv6-plus:` comment, so the line-level
  distinction is visible without opening any doc.

## Original features (category C, implemented so far)

### FR1: Per-process syscall tracing (Phase 1)

A new `trace(mask)` syscall lets any process opt itself (and, by
inheritance, its future children) into having syscalls logged as
`pid: syscall NAME -> RETURN`, filtered by a bitmask of syscall
numbers. A userspace control tool, `xtrace`, wraps this so tracing can
be turned on for an arbitrary command without modifying that command's
source:

```
$ xtrace 65664 echo hi
3: syscall exec -> 2
hi3: syscall write -> 2
3: syscall write -> 1
$ echo untraced
untraced
$
```

(Real transcript from this repo's test suite; `65664` =
`(1<<SYS_exec) | (1<<SYS_write)`.) Full design writeup, safe-scope
rationale, and known limitations: [`docs/tracing.md`](docs/tracing.md).

### FR2/FR3: Per-process accounting + statistics syscall (Phase 2)

Three new counters on every process -- runtime ticks, blocked/wait
ticks, and a total syscall count -- initialized at allocation, reset
on free, deliberately reset (not inherited) across `fork()`, and
preserved across `exec()`. A new `xvstat(idx, &st)` syscall lets any
process read a point-in-time snapshot of any proc-table slot's
counters, memory size, state, and name (index-based enumeration, not
pid-based, per [ADR-0003](docs/decisions/0003-stats-interface.md)).
Every counter write takes the target process's own `p->lock` so
`xvstat(2)` can read `pid`/`state`/`runticks`/`waitticks`/
`syscall_count` as one mutually-consistent snapshot -- see
[ADR-0009](docs/decisions/0009-accounting-counter-locking.md) for the
full locking design, including a lock-order hazard (reusing
`tickslock` for the `waitticks` write) that was considered and
rejected. Full design writeup, fork/exec/exit semantics, and a
`SYS_exit`-counts-itself limitation:
[`docs/accounting.md`](docs/accounting.md).

### FR4: `xvtop` (Phase 3)

A userspace process monitor built entirely on `xvstat(2)`: polls every
proc-table slot, filters out both `UNUSED` and `ZOMBIE` slots (a
zombie is a real, non-`UNUSED` state and needs its own explicit
filter -- invariant #5), sorts by `runticks` descending, and prints a
refreshing table plus an active-process-count summary line.

```
$ xvtop 2 5
=== xvtop refresh 1/2 ===
PID	STATE	SZ	RUNTICKS	WAITTICKS	SYSCALLS	NAME
1	sleep	16384	0	0	23	init
2	sleep	20480	0	5	76	sh
8	run	16384	0	1	31	xvtop
-- 3 active process(es) --
=== xvtop refresh 2/2 ===
PID	STATE	SZ	RUNTICKS	WAITTICKS	SYSCALLS	NAME
1	sleep	16384	0	0	23	init
2	sleep	20480	0	5	76	sh
8	run	16384	0	6	270	xvtop
-- 3 active process(es) --
$
```

(Real transcript from this repo's test suite.) A dedicated regression
test (`tests/xvtop/test_xvtop_zombie_filtered.py`) constructs a
genuine zombie process deterministically and confirms it never appears
as a row. Full design writeup and example transcript:
[`docs/xvtop.md`](docs/xvtop.md).

### FR5: Lottery scheduler experiment (Phase 4)

A second scheduling policy, lottery scheduling, selectable at runtime
alongside the unmodified upstream round-robin baseline via a new
`schedpolicy(policy)` syscall -- not a replacement of it. Each process
gets a configurable ticket count (`settickets(n)`, capped at
`SCHED_MAX_TICKETS` to keep the weighted draw's PRNG modulo from
truncating, [ADR-0013](docs/decisions/0013-ticket-count-bound.md);
inherited across `fork()` like `trace()`'s mask); under lottery
scheduling, every scheduling decision is a weighted random draw
(fixed-seed xorshift32, reproducible by design) over the
currently-runnable set. A process configured with 0 tickets can only
ever run via a deterministic fallback that fires when the whole
`RUNNABLE` set's ticket total hits 0 -- this keeps the *scheduler*
from ever stalling on `RUNNABLE` work (invariant #4), but is not a
guarantee that a 0-ticket process gets scheduled while a nonzero-
ticket competitor stays continuously `RUNNABLE` (see
`docs/scheduler.md`'s "Zero-ticket floor" section for the precise,
corrected statement). Full design writeup, PRNG/locking rationale, and
a captured benchmark comparing the two policies head-to-head on the
same unequal-ticket workload: [`docs/scheduler.md`](docs/scheduler.md).

```
=== baseline (SCHED_RR), tickets 60:20:10:5:60:20:10 ===
runticks: 14 11 12 12 13 13 13        <- roughly equal regardless of tickets

=== lottery (SCHED_LOTTERY), same tickets ===
runticks: 22 17 6 5 22 6 10           <- tracks ticket share (by tier average)
```

(Real transcript from this repo's own build, regenerated directly from
[`benchmarks/raw/scheduler/`](benchmarks/raw/scheduler/) during review
followup -- an earlier hand-typed version of this transcript did not
match those raw files and has been corrected; see
[`benchmarks/methodology.md`](benchmarks/methodology.md) for the full
run and per-tier analysis, independently re-verified line-by-line
against the raw files and correct throughout.)

### FR6: VM extension -- page-fault telemetry (Phase 5)

Per-process counters (`pagefaults`, `pagefaults_failed`) for the
already-upstream lazy-allocation fault path (`sys_sbrk(n,
SBRK_LAZY)`/`vmfault()`), exposed through the existing `xvstat(2)`
interface -- **no new syscall**. This is a deliberately narrow
originality claim: the fault-in mechanism itself predates this
project (see the honesty-layer note in
[`docs/vm-extension.md`](docs/vm-extension.md) and
[ADR-0011](docs/decisions/0011-vm-extension-choice.md)); what's
original is the telemetry, a `kernel/trap.c` diagnostic that
distinguishes an out-of-bounds/out-of-memory failure from a genuine
permission violation (e.g. a write to a process's own read-only text
segment -- `vm_permission_violation()`, added during review followup),
and seven dedicated correctness test programs covering invalid
addresses, permission violations, fork/exec interactions, real memory
exhaustion + reclaim-on-exit, and a direct (not just inferred)
`pagefaults_failed` counter read -- test coverage invariant #6 (VM
isolation) had *none* of before this phase. Full design writeup and
captured real transcripts: [`docs/vm-extension.md`](docs/vm-extension.md).

### FR7: Stress/regression test suite (Phase 6)

Four original stress programs (`user/stress*.c`), one per spec Phase 6
category -- concurrent fork/exit, syscall-heavy, memory-pressure,
scheduler stress -- each racing genuinely concurrent workloads across
every hart (this project's QEMU session boots `-smp 3`) against a
kernel path Phases 1-5 only ever exercised sequentially, plus a
lock/invariant audit against the existing lock-order graph
(`wait_lock -> p->lock`, the leaf `sched_lock`, `kalloc()`'s own leaf
freelist lock). **No kernel control-flow/logic changed and no new
lock was added** -- this phase is pure test/tooling addition against
the already-built Phase 1-5 kernel surface, plus one capacity
constant: `kernel/param.h`'s `FSSIZE` was bumped from upstream's 2000
to 20000 blocks after a real, caught regression (see the "Verify it
yourself" section below and `docs/upstream-delta.md`'s Phase 6 entry
for the full account) -- a build-time size parameter, not a logic
change, and it touches no control-flow path. A clean pass under real
concurrent load is the audit's own evidence that the lock-order graph
established through Phase 5 (ADR-0009/0010/0012) actually holds.
Verified repeatable: the stress category was run three consecutive
times against the same build with no change in between (5/5 passing
each time). Full design writeup and the audit itself:
[`docs/stress-testing.md`](docs/stress-testing.md).

```
$ python3 scripts/run_tests.py --only stress --skip-build
[boot] booting kernel/kernel + fs.img in QEMU ...
[boot] OK: kernel boots, init starts sh, shell runs commands
[test] tests/stress/test_stress_fork_exit.py ...
[test] tests/stress/test_stress_fork_exit.py: PASS
[test] tests/stress/test_stress_scheduler_race.py ...
[test] tests/stress/test_stress_scheduler_race.py: PASS
[test] tests/stress/test_stress_syscalls.py ...
[test] tests/stress/test_stress_syscalls.py: PASS
[test] tests/stress/test_stress_vm_pressure.py ...
[test] tests/stress/test_stress_vm_pressure.py: PASS

============================================================
xv6-plus test summary: 5/5 passed in 4.2s
============================================================
```

(Real transcript from this repo, one of three identical consecutive
passing runs used to verify repeatability.)

### Kernel observability report (Phase 7)

Not a new feature -- a report, using exactly the tools this project
already built (`xtrace`, `xvtop`, `xvstat(2)`) to walk a single real,
captured QEMU session through an end-to-end workload: process
**creation** (a traced `exec()` into `cat`, observed with `xtrace`),
**syscalls** (the same trace, showing `open`/`read`/`write`/`close` as
`cat` streams a real file, including a real, reproduced observation of
console-write non-atomicity between the kernel's own trace-print calls
and the traced process's own output), **scheduling** (a backgrounded
lottery-scheduler competition sampled live, mid-run, through three
`xvtop` refreshes showing each competitor's `runticks`/`selections`
climbing in real time), and **memory** (`vmfaulttest`'s pagefault
telemetry showing lazy growth, first-touch faulting, and no
double-count on re-touch, as a live before/after snapshot). Full
report, with every quoted number copied directly from the raw captured
transcript: [`docs/observability-report.md`](docs/observability-report.md)
(raw session: [`benchmarks/raw/observability/end_to_end_session.txt`](benchmarks/raw/observability/end_to_end_session.txt)).

Not yet built: portfolio polish (Phase 8). The resume narrative in the
spec (§1.7) describes the *finished* project and is not claimed here
yet.

## Roadmap status

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 0 | Reproducible base | Clean xv6 boots; test script runs | **Done** |
| 1 | Syscall tracing foundation | Per-process tracing works without breaking normal execution | **Done** |
| 2 | Process accounting | Tests verify fork/exec/exit behavior and counter monotonicity where appropriate | **Done** |
| 3 | `xvtop` | Tool visibly reports active processes and resource data | **Done** |
| 4 | Scheduler experiment | Benchmark compares fairness/turnaround/response vs baseline scheduler | **Done** |
| 5 | VM extension | Dedicated VM tests + architecture note | **Done** |
| 6 | Stress/race hardening | Repeated stress suite passes | **Done** |
| 7 | Kernel observability report | Report committed | **Done** |
| 8 | Portfolio hardening | -- | Not started |

## Repository layout

```
kernel/, user/, mkfs/, Makefile, README, LICENSE.upstream-xv6   vendored upstream (A) + inline original (C) changes
docs/upstream-delta.md      full A/B/C file-by-file breakdown
docs/decisions/              ADRs for every spec open decision (D1-D7) + ADR-0008 (VCS adaptation) + ADR-0009/0010/0012 (locking/design follow-ups) + ADR-0011 (D5 finalized) + ADR-0013 (ticket-count bound, review-fix followup)
docs/tracing.md              Phase 1 design writeup
docs/accounting.md           Phase 2 design writeup
docs/xvtop.md                Phase 3 design writeup
docs/scheduler.md            Phase 4 design writeup
docs/vm-extension.md         Phase 5 design writeup
docs/stress-testing.md       Phase 6 design writeup + lock/invariant audit
docs/observability-report.md Phase 7 end-to-end observability report
docs/invariants.md           the 8 core invariants (spec §1.5) and their status per phase
docs/toolchain.md            verified toolchain versions, build/debug/test workflow
scripts/run-tests.sh         build + run the full suite (thin wrapper)
scripts/run_tests.py         test harness: build, boot smoke test, discover+run tests/*/test_*.py
scripts/qemu_session.py      QEMU serial-console driver used by the harness and every test module
tests/syscall/                Phase 1 tracing tests
tests/accounting/             Phase 2 process-accounting tests
tests/xvtop/                  Phase 3 xvtop tests
tests/scheduler/              Phase 4 scheduler-experiment tests
tests/vm/                     Phase 5 VM-extension tests
tests/stress/                 Phase 6 stress/race-hardening tests (see tests/stress/README.md)
benchmarks/raw/scheduler/, benchmarks/methodology.md   Phase 4 scheduler-fairness benchmark: raw captured output + analysis
benchmarks/raw/observability/  Phase 7 raw captured end-to-end session transcript
```

## Verify it yourself

Requires a RISC-V GNU toolchain (`riscv64-unknown-elf-gcc` or
equivalent) and `qemu-system-riscv64` >= 7.2 on the PATH; verified
against `riscv64-unknown-elf-gcc` 13.2 and `qemu-system-riscv64` 8.2
on WSL2 Ubuntu (see [`docs/toolchain.md`](docs/toolchain.md)).

```sh
scripts/run-tests.sh --clean
```

Exact captured output from this repo (`make clean && make
kernel/kernel fs.img`, then the full suite):

```
[boot] booting kernel/kernel + fs.img in QEMU ...
[boot] OK: kernel boots, init starts sh, shell runs commands
[test] tests/accounting/test_exec_preserves_counters.py ...
[test] tests/accounting/test_exec_preserves_counters.py: PASS
[test] tests/accounting/test_fork_fresh_counters.py ...
[test] tests/accounting/test_fork_fresh_counters.py: PASS
[test] tests/accounting/test_slot_reuse_reset.py ...
[test] tests/accounting/test_slot_reuse_reset.py: PASS
[test] tests/accounting/test_syscall_count_monotonic.py ...
[test] tests/accounting/test_syscall_count_monotonic.py: PASS
[test] tests/accounting/test_waitticks_from_pause.py ...
[test] tests/accounting/test_waitticks_from_pause.py: PASS
[test] tests/accounting/test_xvstat_bounds.py ...
[test] tests/accounting/test_xvstat_bounds.py: PASS
[test] tests/scheduler/test_lottery_zero_ticket_floor.py ...
[test] tests/scheduler/test_lottery_zero_ticket_floor.py: PASS
[test] tests/scheduler/test_sched_baseline_fairness.py ...
[test] tests/scheduler/test_sched_baseline_fairness.py: PASS
[test] tests/scheduler/test_sched_lottery_fairness.py ...
[test] tests/scheduler/test_sched_lottery_fairness.py: PASS
[test] tests/scheduler/test_schedpolicy_validation.py ...
[test] tests/scheduler/test_schedpolicy_validation.py: PASS
[test] tests/scheduler/test_settickets_validation.py ...
[test] tests/scheduler/test_settickets_validation.py: PASS
[test] tests/scheduler/test_tickets_lifecycle.py ...
[test] tests/scheduler/test_tickets_lifecycle.py: PASS
[test] tests/stress/test_stress_fork_exit.py ...
[test] tests/stress/test_stress_fork_exit.py: PASS
[test] tests/stress/test_stress_scheduler_race.py ...
[test] tests/stress/test_stress_scheduler_race.py: PASS
[test] tests/stress/test_stress_syscalls.py ...
[test] tests/stress/test_stress_syscalls.py: PASS
[test] tests/stress/test_stress_vm_pressure.py ...
[test] tests/stress/test_stress_vm_pressure.py: PASS
[test] tests/syscall/test_no_regression.py ...
[test] tests/syscall/test_no_regression.py: PASS
[test] tests/syscall/test_trace_basic.py ...
[test] tests/syscall/test_trace_basic.py: PASS
[test] tests/syscall/test_trace_fork_inheritance.py ...
[test] tests/syscall/test_trace_fork_inheritance.py: PASS
[test] tests/syscall/test_trace_isolation.py ...
[test] tests/syscall/test_trace_isolation.py: PASS
[test] tests/syscall/test_trace_toggle.py ...
[test] tests/syscall/test_trace_toggle.py: PASS
[test] tests/vm/test_exec_discards_lazy_region.py ...
[test] tests/vm/test_exec_discards_lazy_region.py: PASS
[test] tests/vm/test_fork_lazy_region.py ...
[test] tests/vm/test_fork_lazy_region.py: PASS
[test] tests/vm/test_memory_exhaustion_recovery.py ...
[test] tests/vm/test_memory_exhaustion_recovery.py: PASS
[test] tests/vm/test_oob_access_killed.py ...
[test] tests/vm/test_oob_access_killed.py: PASS
[test] tests/vm/test_pagefault_counting.py ...
[test] tests/vm/test_pagefault_counting.py: PASS
[test] tests/vm/test_pagefault_failed_counted.py ...
[test] tests/vm/test_pagefault_failed_counted.py: PASS
[test] tests/vm/test_permission_fault_killed.py ...
[test] tests/vm/test_permission_fault_killed.py: PASS
[test] tests/xvtop/test_xvtop_basic.py ...
[test] tests/xvtop/test_xvtop_basic.py: PASS
[test] tests/xvtop/test_xvtop_zombie_filtered.py ...
[test] tests/xvtop/test_xvtop_zombie_filtered.py: PASS

============================================================
xv6-plus test summary: 31/31 passed in 23.3s
  [PASS] boot smoke test
  [PASS] tests/accounting/test_exec_preserves_counters.py
  [PASS] tests/accounting/test_fork_fresh_counters.py
  [PASS] tests/accounting/test_slot_reuse_reset.py
  [PASS] tests/accounting/test_syscall_count_monotonic.py
  [PASS] tests/accounting/test_waitticks_from_pause.py
  [PASS] tests/accounting/test_xvstat_bounds.py
  [PASS] tests/scheduler/test_lottery_zero_ticket_floor.py
  [PASS] tests/scheduler/test_sched_baseline_fairness.py
  [PASS] tests/scheduler/test_sched_lottery_fairness.py
  [PASS] tests/scheduler/test_schedpolicy_validation.py
  [PASS] tests/scheduler/test_settickets_validation.py
  [PASS] tests/scheduler/test_tickets_lifecycle.py
  [PASS] tests/stress/test_stress_fork_exit.py
  [PASS] tests/stress/test_stress_scheduler_race.py
  [PASS] tests/stress/test_stress_syscalls.py
  [PASS] tests/stress/test_stress_vm_pressure.py
  [PASS] tests/syscall/test_no_regression.py
  [PASS] tests/syscall/test_trace_basic.py
  [PASS] tests/syscall/test_trace_fork_inheritance.py
  [PASS] tests/syscall/test_trace_isolation.py
  [PASS] tests/syscall/test_trace_toggle.py
  [PASS] tests/vm/test_exec_discards_lazy_region.py
  [PASS] tests/vm/test_fork_lazy_region.py
  [PASS] tests/vm/test_memory_exhaustion_recovery.py
  [PASS] tests/vm/test_oob_access_killed.py
  [PASS] tests/vm/test_pagefault_counting.py
  [PASS] tests/vm/test_pagefault_failed_counted.py
  [PASS] tests/vm/test_permission_fault_killed.py
  [PASS] tests/xvtop/test_xvtop_basic.py
  [PASS] tests/xvtop/test_xvtop_zombie_filtered.py
============================================================
```

(31/31 = the boot smoke test plus 30 test modules -- 4 more than the
27/27 shown for Phase 5, exactly the 4 new `tests/stress/` modules.)

Also verified manually against upstream's own `usertests` regression
suite (not part of the automated harness above -- it takes several
minutes and is a stock upstream program, not an xv6-plus feature under
test), with every number below recomputed directly from three raw,
committed console transcripts under
[`benchmarks/raw/usertests/`](benchmarks/raw/usertests/) (full
precision, `(new-old)/old` convention where a percentage is stated --
these supersede an earlier hand-typed 339.1s/223.5s pair that this
project's own raw-artifact discipline could not verify and has since
replaced):

- **Regression, reproduced** ([`phase6_fssize_regression_repro.txt`](benchmarks/raw/usertests/phase6_fssize_regression_repro.txt)):
  rebuilding this phase's four stress binaries into `UPROGS` with the
  pinned upstream `FSSIZE` (2000) still in place makes `usertests` fail
  in 5.045s at `writebig` with `balloc: out of blocks` -- `SOME TESTS
  FAILED`, the real regression this phase's `FSSIZE` fix (below)
  addresses, not a hypothetical one.
- **Pre-Phase-6 baseline, re-measured** ([`phase5_baseline_pre_phase6.txt`](benchmarks/raw/usertests/phase5_baseline_pre_phase6.txt)):
  the pre-Phase-6 tree (`UPROGS` without the four stress binaries,
  `FSSIZE` 2000) reports `ALL TESTS PASSED` in 184.654s.
- **Post-fix**, this exact tree ([`phase6_fssize_fixed.txt`](benchmarks/raw/usertests/phase6_fssize_fixed.txt)):
  with `FSSIZE` bumped to 20000 and the four stress binaries in
  `UPROGS`, `usertests` reports `ALL TESTS PASSED` in 321.233s -- a
  +73.96% increase over the 184.654s baseline above ((321.233 -
  184.654) / 184.654), and a direct, expected consequence of that same
  `FSSIZE` fix, not a new regression: `usertests`' own
  `diskfull`/`outofinodes` tests deliberately fill the *entire* free
  disk to prove graceful `ENOSPC` handling (both correctly report `OK`
  in the tail below, `balloc: out of blocks`/`ialloc: no inodes` being
  their own expected, intentional output, not a failure), and `FSSIZE`
  growing exactly 10x means those two tests alone now perform roughly
  10x the real disk I/O they did against the smaller pre-Phase-6 image
  -- the work itself is unchanged, only its size is (the other ~90
  `usertests` cases are unaffected by `FSSIZE`, so total runtime scales
  by less than 10x overall; wall time also varies some run-to-run with
  host/QEMU load regardless of `FSSIZE`).

All seven Phase 2/4/5 telemetry fields and their locking are live on
every process throughout, and (Phase 5) `usertests`' own `lazytests`
group exercises the exact `vmfault()` memory-exhaustion path this
project instruments and hardened the diagnostics for. Real captured
tail from the post-fix run
([`phase6_fssize_fixed.txt`](benchmarks/raw/usertests/phase6_fssize_fixed.txt)):

```
test lazy_copy: OK
usertests slow tests starting
test bigdir: OK
test manywrites: OK
test badwrite: OK
test execout: OK
test diskfull: balloc: out of blocks
ialloc: no inodes
ialloc: no inodes
OK
test outofinodes: ialloc: no inodes
OK
ALL TESTS PASSED
$
```

To boot interactively instead: `make qemu` (quit with the QEMU
monitor escape, Ctrl-A x). To debug with gdb: `make qemu-gdb` in one
terminal, `riscv64-unknown-elf-gdb -x .gdbinit` in another.

## Design decisions

Every spec open decision (D1-D7) now has a recorded, **implemented**
ADR under [`docs/decisions/`](docs/decisions/) -- including D5 (VM
extension choice), originally deferred (ADR-0005) pending human input
that never became available mid-build, and finalized in Phase 5
(ADR-0011: page-fault telemetry) the same way D1/D2 were resolved by
implementer default in Phase 0 (ADR-0001/0002) -- plus the one
open-decision process itself had to adapt to this repo's monorepo +
no-git-commands constraints (ADR-0008). Three follow-up ADRs settle
implementation questions the original decisions deliberately deferred
until each feature was actually built: ADR-0009 (Phase 2, the
`syscall_count`/`runticks`/`waitticks` cross-process locking case and
a `tickslock`-reuse hazard that was found and rejected), ADR-0010
(Phase 4, the lottery scheduler's PRNG/two-pass-draw/tickets-lifecycle
design), and ADR-0012 (Phase 5, a second lock-order hazard found and
rejected -- this time in `vmfault()` against `kwait()`'s `copyout()`).
A fourth, ADR-0013, was added during an external review-fix followup
pass (not from a fresh spec decision): the `settickets(2)`
ticket-count upper bound that keeps the lottery draw's PRNG modulo
from truncating a large `RUNNABLE` ticket total.
