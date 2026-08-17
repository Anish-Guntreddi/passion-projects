# Per-process accounting (FR2/FR3, Phase 2)

## What it is

Three new counters on `struct proc` -- runtime ticks, blocked/wait
ticks, and a total syscall count -- plus a syscall-based statistics
interface (`xvstat(2)`, ADR-0003) that lets any process read a
point-in-time snapshot of any proc-table slot's counters, memory size,
state, and name. This is the data Phase 3's `xvtop`
(`docs/xvtop.md`) is built on.

## Kernel side

- **`kernel/proc.h`**: `struct proc` gains three fields:
  - `uint64 runticks` -- ticks this process has spent `RUNNING`
    (ADR-0007: tick granularity, not a finer clock).
  - `uint64 waitticks` -- ticks this process has spent `SLEEPING`
    (blocked/waiting).
  - `uint64 syscall_count` -- total syscalls made since process
    creation.

  Deliberately *not* grouped with Phase 1's `trace_mask` under the
  "owner mutates, no lock" rule: unlike `trace_mask`, these three are
  read cross-process by `xvstat(2)`, so every writer explicitly holds
  `p->lock` around each mutation. See
  `docs/decisions/0009-accounting-counter-locking.md` for the full
  locking reasoning, including a lock-order hazard that rules out
  reusing `tickslock` for the `waitticks` write.

- **`kernel/pstat.h`** (new): `struct xv_pstat`, the fixed-layout
  snapshot struct `xvstat(2)` copies out to userspace -- `pid`,
  `state` (as one of the `XV_*` constants mirroring `enum procstate`),
  `sz`, `runticks`, `waitticks`, `syscalls`, `name`. Included directly
  by both kernel and user code, the same way user code already
  includes `kernel/stat.h` for `struct stat`.

- **`kernel/proc.c`**:
  - `procstat(int idx, struct xv_pstat *out)` -- fills `*out` from
    `proc[idx]` under that slot's own `p->lock`. Returns `-1` if `idx`
    is outside `[0, NPROC)` (the sentinel an enumerating caller uses
    to know it has reached the end of the table); returns `0` and
    fills `*out` (with `pid == 0`, `state == XV_UNUSED`) for a slot
    that is simply not currently in use -- that is not an error, the
    caller decides whether to display it (see "Lifecycle" below and
    `docs/xvtop.md`).
  - Counter lifecycle hooks, described next.

- **`kernel/sysproc.c`**: `sys_xvstat()` reads `idx` (a plain `int`,
  no pointer to validate) and `addr` via `argint()`/`argaddr()`, calls
  `procstat()`, and `copyout()`s the result -- validated by
  `copyout()` itself, same pattern as `sys_fstat()`'s `filestat()`
  (invariant #7). The syscall wrapper stays trivial; all the actual
  field-reading logic lives in the independently-reasoned-about
  `procstat()` helper (handoff brief's "syscalls kept small" rule).

- **`kernel/syscall.c`**: `syscall()` now increments
  `p->syscall_count` (under `p->lock`) for every dispatched syscall,
  *before* calling the handler -- see "Known limitation" below for why
  that ordering matters.

- **`kernel/trap.c`**: `clockintr()` charges one `runtick` (under
  `p->lock`) to whichever process is currently running on that hart,
  on every hart's own periodic timer interrupt.

- **`kernel/proc.c`**'s `sleep()`: adds `(ticks - sleep_start)` to
  `p->waitticks` when a process wakes back up, where `sleep_start` was
  snapshotted (unlocked, deliberately -- ADR-0009) just before the
  process went to sleep.

## Lifecycle: init/fork/exec/exit semantics

- **`allocproc()`** zeroes all three counters on allocation, so a
  freshly-allocated slot starts at zero even before its first fork,
  exec, or syscall (invariant #3).
- **`freeproc()`** zeroes them again on free, so a reused proc-table
  slot never shows a previous occupant's runtime, wait time, or
  syscall count (invariant #5). Tested by
  `tests/accounting/test_slot_reuse_reset.py`.
- **`kfork()`**: deliberately **not** inherited. A child's counters
  stay at the zero `allocproc()` already set -- a child's totals
  describe its own execution from birth, not a copy of whatever the
  parent had already accumulated. (Contrast with Phase 1's
  `trace_mask`, which *is* inherited across `fork()` -- the opposite
  choice, for the opposite reason: tracing is a policy a parent opts
  its whole subtree into, accounting is a per-process fact.) Tested by
  `tests/accounting/test_fork_fresh_counters.py`.
- **`kexec()`**: untouched. Accounting is a per-pid-lifetime total, not
  a per-program-image one -- the same `struct proc`/pid exists before
  and after `exec()`, so its counters (unlike a freshly-exec'd
  process's memory image) carry straight through. Tested by
  `tests/accounting/test_exec_preserves_counters.py`.
- **`kexit()`**: a process's final counter values remain readable via
  `xvstat(2)` while it is a `ZOMBIE` (parent hasn't `wait()`ed yet),
  and are reset to zero the moment `freeproc()` reaps it.

## Known limitation: `SYS_exit` counts itself, but is the last thing counted

`syscall()` increments `syscall_count` *before* dispatching the
handler, specifically because `sys_exit()` -> `kexit()` never returns
to that call site -- it jumps straight into `sched()` and the process
is gone. Counting after dispatch (as Phase 1's trace-line print
does -- see `docs/tracing.md`'s known limitation) would have silently
never counted `exit()` at all. Counting before dispatch means the
`exit` call itself *is* included in the final `syscall_count` a parent
or `xvtop` observes (while the process is briefly a zombie, or via the
snapshot a concurrent `xvstat(2)` catches mid-exit), but it is
necessarily the very last syscall counted for that process.

## Locking

See `docs/decisions/0009-accounting-counter-locking.md` for the full
reasoning. Summary: every write to `runticks`, `waitticks`, and
`syscall_count` takes the target process's own `p->lock`, so
`procstat()` can read all three, plus `pid` and `state`, as one
mutually-consistent snapshot. `sz` and `name` are read in the same
critical section but do not carry that guarantee -- their writers
(`growproc()`, `kexec()`) intentionally keep the pre-existing
lock-free "private to the process" convention (ADR-0006), and this
feature does not change that.

## Test coverage

See `tests/accounting/`:

- `test_slot_reuse_reset.py` -- invariant #5: a proc-table slot reused
  after a previous occupant accumulated syscalls/wait time reports
  fresh (near-zero) counters, not a continuation of the previous
  occupant's totals.
- `test_fork_fresh_counters.py` -- invariant #3: a forked child's
  counters start at zero regardless of how much the parent had already
  accumulated.
- `test_syscall_count_monotonic.py` -- `syscall_count` only ever
  increases within one process's lifetime.
- `test_waitticks_from_pause.py` -- `waitticks` is zero before any
  blocking call and advances by roughly the requested duration after
  `pause(n)`.
- `test_exec_preserves_counters.py` -- invariant #3: counters survive
  `exec()`, continuing to grow rather than resetting.
- `test_xvstat_bounds.py` -- invariant #7: `xvstat(2)` rejects an
  out-of-range index and an invalid user pointer, and keeps working
  normally afterwards.

Run via `scripts/run-tests.sh`.
