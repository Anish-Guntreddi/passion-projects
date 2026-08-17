# xv6-plus

Extended teaching kernel and systems observatory, built on MIT's xv6
RISC-V teaching kernel. Portfolio project 07 of 09 (Track D: OS,
independent of the other projects). Full spec: `07-xv6plus-spec.md`
(portfolio repo root).

**Status: Phases 0-1 of 9 complete.** This is *not* a finished
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

Not yet built: process accounting (Phase 2), `xvtop` (Phase 3), a
scheduler experiment (Phase 4), a VM extension (Phase 5), stress
hardening (Phase 6), an observability report (Phase 7), or portfolio
polish (Phase 8). The resume narrative in the spec (§1.7) describes
the *finished* project and is not claimed here yet.

## Roadmap status

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 0 | Reproducible base | Clean xv6 boots; test script runs | **Done** |
| 1 | Syscall tracing foundation | Per-process tracing works without breaking normal execution | **Done** |
| 2 | Process accounting | -- | Not started |
| 3 | `xvtop` | -- | Not started |
| 4 | Scheduler experiment | -- | Not started |
| 5 | VM extension | -- | Not started |
| 6 | Stress/race hardening | -- | Not started |
| 7 | Kernel observability report | -- | Not started |
| 8 | Portfolio hardening | -- | Not started |

## Repository layout

```
kernel/, user/, mkfs/, Makefile, README, LICENSE.upstream-xv6   vendored upstream (A) + inline original (C) changes
docs/upstream-delta.md      full A/B/C file-by-file breakdown
docs/decisions/              ADRs for every spec open decision (D1-D7) + ADR-0008 (VCS adaptation)
docs/tracing.md              Phase 1 design writeup
docs/invariants.md           the 8 core invariants (spec §1.5) and their status per phase
docs/toolchain.md            verified toolchain versions, build/debug/test workflow
scripts/run-tests.sh         build + run the full suite (thin wrapper)
scripts/run_tests.py         test harness: build, boot smoke test, discover+run tests/*/test_*.py
scripts/qemu_session.py      QEMU serial-console driver used by the harness and every test module
tests/syscall/               Phase 1 tracing tests
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

============================================================
xv6-plus test summary: 6/6 passed in 1.7s
  [PASS] boot smoke test
  [PASS] tests/syscall/test_no_regression.py
  [PASS] tests/syscall/test_trace_basic.py
  [PASS] tests/syscall/test_trace_fork_inheritance.py
  [PASS] tests/syscall/test_trace_isolation.py
  [PASS] tests/syscall/test_trace_toggle.py
============================================================
```

To boot interactively instead: `make qemu` (quit with the QEMU
monitor escape, Ctrl-A x). To debug with gdb: `make qemu-gdb` in one
terminal, `riscv64-unknown-elf-gdb -x .gdbinit` in another.

## Design decisions

Every spec open decision (D1-D7) has a recorded ADR under
[`docs/decisions/`](docs/decisions/), including the ones deferred to
later phases (D5, VM extension choice) and the one open-decision
process itself had to adapt to this repo's monorepo + no-git-commands
constraints (ADR-0008).
