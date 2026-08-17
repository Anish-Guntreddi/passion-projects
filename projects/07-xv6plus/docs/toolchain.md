# Toolchain and QEMU workflow

## Verified environment

Built and tested on WSL2 Ubuntu, from a Windows 11 host, via
`wsl -d Ubuntu -- bash -lc '<cmd>'`. Versions actually used for the
Phase 0/1 build and test run (see the exact test command's output in
the project's build log):

| Tool | Version | Role |
|---|---|---|
| `riscv64-unknown-elf-gcc` | 13.2 | Cross-compiler for `kernel/` and `user/` (`TOOLPREFIX=riscv64-unknown-elf-`, auto-detected by `Makefile`) |
| `qemu-system-riscv64` | 8.2 | Emulates the `virt` RISC-V machine xv6 boots on |
| GNU `make` | (system) | Build orchestration (`Makefile`, vendored + extended, see `docs/upstream-delta.md`) |
| host `gcc` | 13.3 | Builds `mkfs/mkfs`, the host-side tool that packs `fs.img` |
| `python3` | 3.12 | Test harness (`scripts/run_tests.py`), stdlib only, no venv needed |

`MIN_QEMU_VERSION` in `Makefile` is 7.2; 8.2 is well within range.

## Building

```sh
make kernel/kernel fs.img   # what scripts/run_tests.py actually invokes
# or, for an interactive session:
make qemu                   # runs qemu-system-riscv64 in the foreground;
                             # quit with the QEMU monitor escape, Ctrl-A x
```

`make` alone only builds `kernel/kernel` (the first Makefile target,
hence `make`'s default goal) -- it does *not* build `fs.img`, which has
its own separate target. The test harness (below) builds both
explicitly rather than relying on the default goal.

## Debugging (quality bar, spec §1.8: gdb/QEMU, not printf-only)

```sh
make qemu-gdb          # terminal 1: boots QEMU paused, waiting for gdb
riscv64-unknown-elf-gdb -x .gdbinit   # terminal 2
```

`.gdbinit` is generated from the vendored `.gdbinit.tmpl-riscv` (the
template substitutes in a per-UID GDB port so concurrent users don't
collide); both are unmodified upstream. This path was not exercised
in Phase 0/1 -- the automated test harness never hit a kernel bug that
needed interactive debugging -- but is verified present and wired
correctly through the vendored `Makefile`.

## Why the test harness doesn't use `make qemu`

`make qemu` runs `qemu-system-riscv64` in the foreground and expects a
human to quit it with the QEMU monitor escape (Ctrl-A x). That is easy
for a person at a real terminal and awkward to drive reliably from a
non-interactive script. `scripts/qemu_session.py` instead launches
`qemu-system-riscv64` directly with the same flags `make qemu` would
pass (see `QEMUOPTS` in `Makefile`), as a child process the harness
owns outright and simply terminates when a test is done -- no reliance
on sending an escape byte sequence through a pipe.

## Test command

```sh
scripts/run-tests.sh            # build (incremental) + full suite
scripts/run-tests.sh --clean    # make clean first, then build + suite
scripts/run-tests.sh --only syscall   # just tests/syscall/*
```

See `scripts/run_tests.py`'s module docstring for the stage-by-stage
breakdown (build -> boot smoke test -> discovered `tests/*/test_*.py`
modules -> summary), and the project README for a captured run.
