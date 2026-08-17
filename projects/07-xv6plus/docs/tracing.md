# Syscall tracing (FR1, Phase 1)

## What it is

A per-process syscall tracing framework: any process can opt itself
(and, by inheritance, its future children) into having its syscalls
logged to the console as `pid: syscall NAME -> RETURN`, filtered by a
bitmask of which syscall numbers to log.

## Kernel side

- **`kernel/proc.h`**: `struct proc` gains `int trace_mask`. Bit *i*
  set means syscall number *i* (from `kernel/syscall.h`) is traced for
  that process.
- **`kernel/sysproc.c`**: `sys_trace()` implements the new syscall --
  reads one `int` argument via `argint()` and sets
  `myproc()->trace_mask = mask`. That's the entire implementation:
  it only ever touches the calling process's own struct proc.
- **`kernel/syscall.c`**: `syscall()` (the trap-time dispatcher) now,
  after invoking the handler and storing its return value in
  `p->trapframe->a0`, checks `(p->trace_mask >> num) & 1` and, if set,
  prints `"%d: syscall %s -> %d\n"` using a new `syscall_names[]`
  table that mirrors the existing `syscalls[]` dispatch table.
- **Lifecycle** (`kernel/proc.c`): `allocproc()` zeroes `trace_mask` on
  allocation; `freeproc()` zeroes it again on free (belt-and-suspenders
  -- see the invariant #3/#5 discussion in `docs/invariants.md`);
  `kfork()` copies the parent's `trace_mask` into the child. `kexec()`
  is untouched, so tracing survives `exec()` for free -- the struct
  proc doing the tracing is the same one before and after exec, only
  its page table/registers/name change.

## "Safe scope": what is deliberately *not* captured

FR1 asks for "syscall number/name/return data... within safe scope."
This implementation captures exactly the number, the name, and the
return value -- **not** the syscall's arguments. Printing arguments
generically would mean, for every syscall, correctly re-deriving
which of its `a0`-`a5` slots are integers vs. pointers vs.
NUL-terminated user strings, and safely `copyin`-ing the pointer ones
-- a much larger, syscall-specific surface that is easy to get subtly
wrong (a bad length, an unchecked pointer) in exactly the kind of code
path invariant #7 exists to protect. Skipping argument capture keeps
the *entire* tracing implementation to five small, obviously-safe
touch points (see `docs/upstream-delta.md`), which is the actual
point of "safe scope."

## Userspace side

- **`user/xtrace.c`** -- the control tool required by FR1. Usage:

  ```
  xtrace mask command [args...]
  ```

  It calls `trace(mask)` on itself, then `exec()`s straight into
  `command`. Because `trace_mask` survives `exec()` and is inherited
  by `fork()`, `command` (and any children it forks) run traced from
  their very first syscall onward, until/unless something in that
  process tree calls `trace(0)` itself.

- **`user/tracetest.c`** -- a small, self-contained program used only
  by the test suite (`tests/syscall/test_trace_toggle.py`) to prove
  `trace(0)` really disables tracing for a *still-running* process,
  not just "until the next exec": it calls `trace(1<<SYS_getpid)`,
  calls `getpid()` once (traced), calls `trace(0)`, calls `getpid()`
  again (not traced).

## Example transcript

Captured from a real `scripts/run-tests.sh` session (mask `65664` =
`(1<<SYS_exec) | (1<<SYS_write)`, `SYS_exec=7`, `SYS_write=16`):

```
$ xtrace 65664 echo hi
3: syscall exec -> 2
hi3: syscall write -> 2
3: syscall write -> 1
$ echo untraced
untraced
$
```

Reading this: `xtrace` execs into `echo hi` (pid 3); the traced
`exec` call is logged with its return value (`argc == 2`); `echo`
makes two `write()` calls (one for `"hi"`, one for the trailing
newline), both logged; the very next command, run without `xtrace`,
produces no trace output at all -- tracing is strictly per-process.

## Test coverage

See `tests/syscall/`: `test_trace_basic.py` (number/name/return
capture, doesn't corrupt the traced program's own output),
`test_trace_isolation.py` (an untraced command right after a traced
one stays silent), `test_trace_fork_inheritance.py` (tracing survives
dozens of `fork()` calls under `forktest`'s proc-table-filling load),
`test_trace_toggle.py` (`trace(0)` really disables tracing
mid-process), `test_no_regression.py` (a session that never calls
`trace()` behaves exactly like stock xv6). Run with
`scripts/run-tests.sh`.

## Known limitation

`SYS_exit` is never traced -- see `docs/invariants.md`, "Known,
deliberate limitation."
