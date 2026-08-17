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
