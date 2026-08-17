# xv6-plus

Extended teaching kernel and systems observatory, built on MIT's xv6
RISC-V teaching kernel. Portfolio project 07 of 09 (Track D: OS,
independent of the other projects). Full spec: `07-xv6plus-spec.md`
(portfolio repo root).

**Status: Phases 0-3 of 9 complete.** This is *not* a finished
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

Not yet built: a scheduler experiment (Phase 4), a VM extension
(Phase 5), stress hardening (Phase 6), an observability report (Phase
7), or portfolio polish (Phase 8). The resume narrative in the spec
(§1.7) describes the *finished* project and is not claimed here yet.

## Roadmap status

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 0 | Reproducible base | Clean xv6 boots; test script runs | **Done** |
| 1 | Syscall tracing foundation | Per-process tracing works without breaking normal execution | **Done** |
| 2 | Process accounting | Tests verify fork/exec/exit behavior and counter monotonicity where appropriate | **Done** |
| 3 | `xvtop` | Tool visibly reports active processes and resource data | **Done** |
| 4 | Scheduler experiment | -- | Not started |
| 5 | VM extension | -- | Not started |
| 6 | Stress/race hardening | -- | Not started |
| 7 | Kernel observability report | -- | Not started |
| 8 | Portfolio hardening | -- | Not started |

## Repository layout

```
kernel/, user/, mkfs/, Makefile, README, LICENSE.upstream-xv6   vendored upstream (A) + inline original (C) changes
docs/upstream-delta.md      full A/B/C file-by-file breakdown
docs/decisions/              ADRs for every spec open decision (D1-D7) + ADR-0008 (VCS adaptation) + ADR-0009 (Phase 2 locking follow-up)
docs/tracing.md              Phase 1 design writeup
docs/accounting.md           Phase 2 design writeup
docs/xvtop.md                Phase 3 design writeup
docs/invariants.md           the 8 core invariants (spec §1.5) and their status per phase
docs/toolchain.md            verified toolchain versions, build/debug/test workflow
scripts/run-tests.sh         build + run the full suite (thin wrapper)
scripts/run_tests.py         test harness: build, boot smoke test, discover+run tests/*/test_*.py
scripts/qemu_session.py      QEMU serial-console driver used by the harness and every test module
tests/syscall/                Phase 1 tracing tests
tests/accounting/             Phase 2 process-accounting tests
tests/xvtop/                  Phase 3 xvtop tests
tests/scheduler/, tests/vm/, tests/stress/   reserved for later phases (see per-directory README.md)
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
[test] tests/xvtop/test_xvtop_basic.py ...
[test] tests/xvtop/test_xvtop_basic.py: PASS
[test] tests/xvtop/test_xvtop_zombie_filtered.py ...
[test] tests/xvtop/test_xvtop_zombie_filtered.py: PASS

============================================================
xv6-plus test summary: 14/14 passed in 5.9s
  [PASS] boot smoke test
  [PASS] tests/accounting/test_exec_preserves_counters.py
  [PASS] tests/accounting/test_fork_fresh_counters.py
  [PASS] tests/accounting/test_slot_reuse_reset.py
  [PASS] tests/accounting/test_syscall_count_monotonic.py
  [PASS] tests/accounting/test_waitticks_from_pause.py
  [PASS] tests/accounting/test_xvstat_bounds.py
  [PASS] tests/syscall/test_no_regression.py
  [PASS] tests/syscall/test_trace_basic.py
  [PASS] tests/syscall/test_trace_fork_inheritance.py
  [PASS] tests/syscall/test_trace_isolation.py
  [PASS] tests/syscall/test_trace_toggle.py
  [PASS] tests/xvtop/test_xvtop_basic.py
  [PASS] tests/xvtop/test_xvtop_zombie_filtered.py
============================================================
```

Also verified manually against upstream's own `usertests` regression
suite (not part of the automated harness above -- it takes several
minutes and is a stock upstream program, not an xv6-plus feature under
test): `usertests` reports `ALL TESTS PASSED` against this tree, with
the Phase 2 accounting fields and their `p->lock` discipline live on
every process throughout.

To boot interactively instead: `make qemu` (quit with the QEMU
monitor escape, Ctrl-A x). To debug with gdb: `make qemu-gdb` in one
terminal, `riscv64-unknown-elf-gdb -x .gdbinit` in another.

## Design decisions

Every spec open decision (D1-D7) has a recorded ADR under
[`docs/decisions/`](docs/decisions/), including the ones deferred to
later phases (D5, VM extension choice) and the one open-decision
process itself had to adapt to this repo's monorepo + no-git-commands
constraints (ADR-0008). ADR-0009 is a Phase 2 follow-up to D6/ADR-0006,
settling the specific cross-process-read locking case (and a
`tickslock`-reuse hazard) that arose implementing the accounting
counters.
