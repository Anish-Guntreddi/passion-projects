# Scheduler experiment: lottery scheduling (FR5, Phase 4)

## What it is

A second scheduling policy, lottery scheduling, selectable at runtime
alongside upstream's unmodified round-robin baseline via a new
`schedpolicy(2)` syscall -- not a replacement of the baseline (ADR-0004,
ADR-0010). Each `RUNNABLE` process gets a configurable *ticket* count
(`settickets(2)`); under `SCHED_LOTTERY`, every scheduling decision is
a weighted random draw over the currently-`RUNNABLE` set, so a
process's expected CPU share is proportional to its ticket share of
the total.

## Kernel side

- **`kernel/sched.h`** (new): `SCHED_RR` (0, default) / `SCHED_LOTTERY`
  (1) policy constants and `SCHED_DEFAULT_TICKETS` (10), shared by
  kernel and user code (same convention as `kernel/vm.h`'s
  `SBRK_EAGER`/`SBRK_LAZY`).
- **`kernel/proc.h`**: `struct proc` gains `int tickets` (this
  process's ticket count) and `uint64 sched_selections` (how many
  times the scheduler has chosen this process).
- **`kernel/proc.c`**:
  - `sched_policy` (global) and `sched_lock` + `rng_state` (a
    fixed-seed xorshift32 PRNG) -- see ADR-0010 for why the seed is
    fixed, not entropy-derived.
  - `schedule_roundrobin()` -- upstream's scheduling loop, unchanged
    except for the `sched_selections++` instrumentation (factored
    through a shared `run_selected()` helper both policies call).
  - `schedule_lottery()` -- one weighted draw and one selection per
    call, with a bounded retry against a two-pass race (ADR-0010) and
    a deterministic `pick_first_runnable()` fallback whenever the
    ticket total is 0 (this is also the zero-ticket-floor mechanism).
  - `sched_settickets(int)` / `sched_setpolicy(int)` -- the actual
    validation/locking logic behind the two new syscalls below.
  - `procstat()` extended to fill `tickets`/`selections` into
    `struct xv_pstat`.
- **`kernel/sysproc.c`**: `sys_settickets()` / `sys_schedpolicy()` --
  thin wrappers, per the handoff brief's "syscalls kept small" rule.
- **`kernel/syscall.h`/`.c`**: `SYS_settickets` (24), `SYS_schedpolicy`
  (25).
- **`kernel/pstat.h`**: `struct xv_pstat` gains `tickets`/`selections`.

## Syscalls

- **`int settickets(int n)`** -- sets the calling process's own ticket
  count. `n < 0` is rejected (-1); `n == 0` is accepted (a
  deliberately low-priority configuration, not an error -- see
  "Zero-ticket floor" below); `n > SCHED_MAX_TICKETS` (100000,
  `kernel/sched.h`) is also rejected (-1) -- added during review
  followup, see
  [`docs/decisions/0013-ticket-count-bound.md`](decisions/0013-ticket-count-bound.md):
  without a bound, a large enough sum of `RUNNABLE` tickets would
  truncate through `draw_and_run()`'s uint32 PRNG modulo
  (`winner = lottery_rand() % total`), making some processes
  structurally unreachable by any draw. Takes `p->lock` on write
  (ADR-0006 case (b): tickets is read cross-process by the scheduler,
  not just monitored).
- **`int schedpolicy(int policy)`** -- sets the global scheduling
  policy. Only `SCHED_RR`/`SCHED_LOTTERY` are accepted; anything else
  is rejected (-1) and leaves the current policy unchanged. Returns
  the *previous* policy on success, so a benchmark harness can restore
  it afterward (see `user/schedbench.c`).

## Lifecycle: init/fork/exec/exit semantics

- **`allocproc()`**: `tickets` defaults to `SCHED_DEFAULT_TICKETS`
  (10, not 0 -- a never-configured process still gets a normal share);
  `sched_selections` starts at 0.
- **`freeproc()`**: both reset to 0 -- invariant #5, same pattern as
  every earlier phase's fields. Tested by
  `tests/scheduler/test_tickets_lifecycle.py`.
- **`kfork()`**: `tickets` is **inherited** (copied parent-to-child,
  like Phase 1's `trace_mask` -- a forked workload keeps its parent's
  configured share). `sched_selections` is **not** inherited (like
  Phase 2's counters -- a child's own selection history starts at
  birth). See ADR-0010 for the rationale.
- **`kexec()`**: untouched -- both fields are per-pid-lifetime, same
  as the Phase 2 counters.

## Zero-ticket floor (invariant #4) -- what it guarantees and what it doesn't

**Precise claim (fixed during review followup -- the original wording
here overstated this):** `schedule_lottery()`'s `pick_first_runnable()`
fallback triggers only when the currently-`RUNNABLE` set's ticket
*total* is exactly 0 -- either because there is no `RUNNABLE` process
at all, or because *every* `RUNNABLE` process currently has 0 tickets.
Only in that specific window does a 0-ticket process get picked (in
table order, like one round-robin step). This is what keeps invariant
#4 ("the scheduler always eventually selects eligible work") true: the
lottery policy can never stall with `RUNNABLE` work sitting unpicked.

**What this does *not* guarantee: a general starvation bound against
sustained competition.** `draw_and_run()`'s weighted draw
(`winner = lottery_rand() % total`, `cum += p->tickets`) gives a
0-ticket process's cumulative range zero width -- it structurally
cannot ever be the winner of a weighted draw, by design (0 tickets
means 0 share of the lottery, exactly as a proportional-share
scheduler should behave). So the fallback is the *only* way a
0-ticket process ever runs, and the fallback only fires when the
`RUNNABLE` ticket total is 0. If even one nonzero-ticket process
stays continuously `RUNNABLE` (e.g. a CPU-bound loop that never
blocks), the total never drops to 0, and a 0-ticket process competing
against it is never selected -- not "rare," but never, for as long as
that condition holds. This is a known, deliberate limitation, not a
hard anti-starvation bound: a real "compensation tickets"-style
mechanism (aging a process's effective ticket count the longer it
goes unselected) would be needed to close this gap, and this project
does not implement one. See
[`docs/decisions/0010-lottery-scheduler-design.md`](decisions/0010-lottery-scheduler-design.md)
decision 5 and `tests/scheduler/test_lottery_zero_ticket_floor.py`,
whose workload (four 0-ticket children alongside three *finite*,
wall-clock-bounded 30-ticket children that eventually all exit,
driving the total to 0) is exactly the condition this floor covers --
not a persistently-competing workload.

## Benchmark: `schedbench` and workload calibration

`user/schedbench.c` sets a policy, forks one child per ticket count
given on the command line, and has each child spin on pure CPU work
(no blocking syscalls in the loop) until `TARGET_TICKS` (30, ~3
seconds) have passed since the whole benchmark's own start, then
self-reports its `runticks`/`waitticks`/`selections` snapshot.

**Why wall-clock-bounded, not a fixed iteration count.** An earlier
version spun a fixed number of arithmetic iterations. Calibration
found that count complete in under one timer tick on this
environment's QEMU/host combination -- nowhere near enough scheduling
decisions to measure fairness. "Run until N wall ticks pass" instead
gives every run the same fixed *competition window* regardless of
host throughput; the fairness signal is in how much of that fixed
window each child's `runticks`/`selections` capture, not in
`elapsed` (which converges to roughly `TARGET_TICKS` for every child,
including a starved one, by construction).

**Why each child gets its own pipe instead of printing straight to
the console.** Up to `NCPU` (this project's QEMU session boots
`-smp 3`) children genuinely run concurrently. This pinned revision's
`printf()`/`fprintf()` (`user/printf.c`) write one byte per `write(2)`
syscall with no per-line atomicity; two processes both mid-`printf()`
at once produced a byte-interleaved, unparseable mess on the shared
console during this feature's own calibration runs (observed directly
while tuning `TARGET_TICKS`/`BURST_ITERS` below, before the pipe
design was added -- not preserved as a file anywhere, since it was a
calibration artifact, not a deliverable). Giving each
child its own pipe makes cross-process interleaving impossible (only
that one child ever writes to it); only the single-threaded parent
ever writes to the real console, after collecting each child's report
in turn.

**Shell argument limit.** `user/sh.c`'s `MAXARGS` is 10 (program name
+ 9 words), not the kernel's `MAXARG` (32) -- `schedbench policy
tickets...` invoked from an interactive/scripted shell command line is
capped at 7 ticket values.

**`MAX_CHILDREN` (direct, non-shell callers).** Fixed during review
followup: this was originally `MAXARG - 2` (30), which overstated the
program's real capacity by more than 2x. The parent keeps one pipe
read-fd open per still-uncollected child (the write end is closed
immediately, but the read end is retained until that child's report is
drained -- see "own pipe" above), on top of the 3 fds a process
already holds for stdin/stdout/stderr, all bounded by `NOFILE`
(`kernel/param.h`, 16). Forking child `i` (0-indexed) needs 2 free fd
slots for its `pipe()` call while `i` earlier read-ends are still
retained: `3 + i + 2 <= NOFILE`, so the largest `i` that can succeed
is `NOFILE - 6`, i.e. `NOFILE - 4` total children. `MAX_CHILDREN` is
now defined as `NOFILE - 4` directly (12, with this project's
`NOFILE=16`) instead of a hardcoded, wrong constant. Verified directly
against a real build: a throwaway probe program performing exactly
`schedbench`'s pipe-then-retain-read-end pattern reported `pipe
failed at i=12 (0-indexed) -- 12 children succeeded` -- i.e. children
0-11 (12 total) succeed, child 12 (the 13th) is the first to fail,
exactly matching `NOFILE - 4 = 12`. (This bound is still far above
what the shell caps real usage at, 7 ticket values, per the paragraph
above -- it only matters for a future direct caller.)

## Captured real run

From this repository's own build, `qemu-system-riscv64` 8.2, `-smp 3`,
via `scripts/qemu_session.py` (not fabricated -- see the raw
`schedbench: report ...` lines, one per child, and cross-check against
`benchmarks/raw/scheduler/`):

(Fixed during review followup: the numbers below were previously
hand-typed and did not match this repository's own
`benchmarks/raw/scheduler/*.txt` captures -- a transcription error,
not a re-run. They are now regenerated directly from those raw files,
identical to `benchmarks/methodology.md`'s tables, which were
independently re-verified against the raw files line-by-line and were
correct throughout.)

**Baseline (`SCHED_RR`), unequal tickets `60 20 10 5 60 20 10`:**

```
tickets=60 runticks=14   tickets=20 runticks=11   tickets=10 runticks=12
tickets=5  runticks=12   tickets=60 runticks=13   tickets=20 runticks=13
tickets=10 runticks=13
```

Range: 11-14 runticks. A 12x ticket spread produces only a 1.27x
runticks spread -- round-robin ignores tickets entirely, as expected
(`schedule_roundrobin()` never reads `p->tickets`).

**Lottery (`SCHED_LOTTERY`), same ticket set:**

```
tickets=60 runticks=22   tickets=20 runticks=17   tickets=10 runticks=6
tickets=5  runticks=5    tickets=60 runticks=22   tickets=20 runticks=6
tickets=10 runticks=10
```

Per-ticket-tier *averages* are monotonic (60-ticket: 22.0, 20-ticket:
11.5, 10-ticket: 8.0, 5-ticket: 5.0) -- a 4.4x spread (22 vs. 5) and a
clear directional fairness signal the baseline above does not show.
Individual runs are noisier than the tier averages suggest, though:
one 20-ticket child scored only 6 runticks, *below* a 10-ticket
child's 10 -- a reminder that a handful of draws per process (a few
dozen selections total) is nowhere near enough to remove sampling
noise, only enough to show the right direction on average. (Not exact
proportionality -- lottery scheduling is intentionally probabilistic
at this sample size; see `tests/scheduler/test_sched_lottery_fairness.py`
for the exact, generous tolerances this data was used to choose.)

**Lottery, zero-ticket floor, tickets `0 0 30 30 30 0 0`:**

```
tickets=0  runticks=1    tickets=0  runticks=1    tickets=30 runticks=27
tickets=30 runticks=29   tickets=30 runticks=26   tickets=0  runticks=2
tickets=0  runticks=2
```

0-ticket children: 1-2 runticks each (average 1.5). 30-ticket
children: 26-29 each (average 27.33) -- an 18.2x ratio between
averages. Every one of the 7 processes still completed and was reaped
(no hang). Per the "what it guarantees and what it doesn't" discussion
above: this specific workload's 30-ticket children are themselves
finite (wall-clock-bounded, so they eventually stop being `RUNNABLE`
and the total ticket count drops to 0, which is exactly the condition
the zero-ticket floor requires) -- this run demonstrates that
condition, not a bound against a persistently-competing nonzero-ticket
workload.

## Test coverage

See `tests/scheduler/`:

- `test_tickets_lifecycle.py` -- invariants #3/#5: tickets inherited
  across `fork()`, reset to the default on free (not a stale prior
  value).
- `test_settickets_validation.py` -- argument validation (negative
  rejected, 0 and positive accepted).
- `test_schedpolicy_validation.py` -- argument validation, plus an
  invariant #8 regression check.
- `test_sched_baseline_fairness.py` -- `SCHED_RR` ignores tickets.
- `test_sched_lottery_fairness.py` -- `SCHED_LOTTERY` differentiates
  by ticket share.
- `test_lottery_zero_ticket_floor.py` -- invariant #4's starvation
  bound.

Run via `scripts/run-tests.sh`. Also re-verified against upstream's
own `usertests` regression suite (`ALL TESTS PASSED`, 223.5s -- re-run
against this exact tree during review followup; timing varies a
little run-to-run with host/QEMU load) with the Phase 4 fields, locks,
and scheduler dispatch live for every process throughout (the baseline
`SCHED_RR` policy the whole run, since `usertests` never calls
`schedpolicy()`).
