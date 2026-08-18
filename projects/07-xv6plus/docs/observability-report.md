# Kernel observability report (Phase 7)

## Purpose (spec §Part 3 roadmap row 7)

> Use tracing/xvtop to explain an end-to-end workload (creation,
> syscalls, scheduling, memory). Exit criterion: report committed.

This is a report, not a new kernel feature -- Phase 7 adds no new
kernel or userspace source. It exists to answer one question with
real, captured evidence rather than a description of what the Phases
1-5 instrumentation is *supposed* to show: **if you actually run a
workload through this kernel and watch it with the tools this project
built (`xtrace`, `xvtop`, `xvstat(2)`), what do you actually see, at
every stage from a process being created to it exiting?**

## Methodology

Everything below is one real, unedited console transcript from a
single QEMU session of this exact tree (3 harts, 128MB,
`kernel/kernel` + `fs.img` built from a clean `make`), captured by
[`scripts/capture_observability.py`](../scripts/capture_observability.py)
-- built on `scripts/qemu_session.py`, the same driver
`scripts/run_tests.py` uses for every automated test, and following
the same raw-artifact-only convention `scripts/benchmark.py` already
established for `benchmarks/raw/scheduler/`: the script writes the
transcript, `benchmarks/raw/observability/end_to_end_session.txt`
below, and computes/asserts nothing itself -- every number quoted in
this report is copied directly from that file, not computed here and
not hand-typed:
[`benchmarks/raw/observability/end_to_end_session.txt`](../benchmarks/raw/observability/end_to_end_session.txt).
This capture is a one-shot documentation artifact, not part of the
automated `tests/<category>/` suite `scripts/run_tests.py` discovers
(there is nothing to assert here beyond a final plain-shell regression
check). Reproduce it yourself with:

```sh
scripts/run-tests.sh --skip-build   # build first if needed
python3 scripts/capture_observability.py
```

The workload walks through the same four stages spec §1.7's deliverable
list names: process **creation**, **syscalls**, **scheduling**, and
**memory** -- in that order, using `xvtop` (Phase 3) to bracket the
whole thing (a snapshot before and after) and `xtrace` (Phase 1),
a backgrounded scheduler benchmark (Phase 4's `schedbench`, observed
live through `xvtop`), and a memory instrumentation program (Phase
5's `vmfaulttest`) for the three stages in between.

## Stage 1 -- baseline: what "creation" looks like before anything runs

```
$ xvtop 1 5
=== xvtop refresh 1/1 ===
PID	STATE	SZ	RUNTICKS	WAITTICKS	SYSCALLS	NAME
1	sleep	16384	0	0	23	init
2	sleep	20480	0	0	16	sh
3	run	20480	0	0	31	xvtop
-- 3 active process(es) --
```

Three processes: `init` (pid 1, the first process, `sleep`ing in
`wait()` for orphans), the interactive shell `sh` (pid 2, `sleep`ing
for the next line of input), and `xvtop` itself (pid 3, `run`ning --
naturally, since it's the one process actively producing this output).
Every field here is exactly the Phase 2/3/4/5 telemetry surface
(`struct xv_pstat`, `kernel/pstat.h`): `SZ` is memory size in bytes,
`RUNTICKS`/`WAITTICKS` are ticks accounted `RUNNING`/`SLEEPING`,
`SYSCALLS` is the lifetime syscall count. `xvtop`'s own pid (3) is a
live example of invariant #3 (telemetry initialized at allocation): it
was `fork()`+`exec()`'d by `sh` moments before this line printed, and
its counters already reflect real activity from its own startup
syscalls, not stale or garbage values.

## Stage 2 -- syscalls: `xtrace` on a real file-reading command

```
$ xtrace 2195616 cat README
4: syscall exec -> 2
4: syscall open -> 3
4: syscall read -> 512
xv6 is a re-implementation of Dennis Ritchie's and Ken Thompson's Unix
Version 6 (v6). ...
```

`2195616` = `(1<<SYS_exec) | (1<<SYS_open) | (1<<SYS_read) |
(1<<SYS_write) | (1<<SYS_close)` (`kernel/syscall.h`): `sh` forks a
new process (pid 4, untraced -- the fork happens before `trace()` is
ever called, exactly as designed: tracing is opt-in from the point
`trace()` is called forward, not retroactive), that process calls
`trace(2195616)` on itself, then `exec()`s into `xtrace`'s own image
replaced by `cat`. Because `trace_mask` lives on `struct proc` and
`kexec()` never touches it (`docs/tracing.md`), the *exec into `cat`
itself* is the first traced line (`4: syscall exec -> 2`, return value
2 = `cat`'s own initial fd/argv setup completing) -- concretely
demonstrating that tracing survives the exact "process creation via
image replacement" event the spec's "creation" bullet asks to be
shown, not just fork. Every one of `cat`'s own `open`/`read`/`write`/
`close` calls against `README` then appears in order as it streams the
file to the console.

**A real, reproduced observability finding, not a bug:** the full raw
transcript (see the linked file) shows several trace lines
(`4: syscall write -> 512`) printed in the *middle* of `README`'s own
text, mid-word. This is not a syscall-ordering error -- the dispatch
order in `kernel/syscall.c` is strictly write-then-trace-print, so the
512 bytes genuinely were already handed to the console driver before
the trace line is printed -- it is the shared UART character device
draining the just-written 512 bytes at roughly the same real
wall-clock rate the next kernel `printf()` call is also emitting
characters, with no line-buffering on either side. It is the *same*
single-writer-at-a-time-but-no-atomicity hazard `docs/scheduler.md`
and every Phase 6 stress-program comment document for *multiple
concurrent processes* sharing the console -- here it shows up even
for a *single* traced process, because the kernel's own trace-print
call and that process's `write(2)` are two logically separate writers
to the same character stream. `xvtop`'s output (Stage 1/5, printed a
whole table per `write(2)` call from a single quiescent process with
no competing writer) never shows this; it is specific to tracing a
process that is itself actively streaming output.

## Stage 3 -- scheduling: a live, running competition observed through `xvtop`

A backgrounded `schedbench` run (Phase 4's scheduler-experiment
driver, `user/schedbench.c`) under `SCHED_LOTTERY`, four children with
tickets 60/20/10/5, sampled mid-run by three `xvtop` refreshes:

```
$ schedbench 1 60 20 10 5 &
$ xvtop 3 6
=== xvtop refresh 1/3 ===
PID	STATE	SZ	RUNTICKS	WAITTICKS	SYSCALLS	NAME
8	runble	16384	2	0	762	schedbench
11	runble	16384	2	0	742	schedbench
7	run	20480	1	0	31	xvtop
6	sleep	16384	1	0	16	schedbench
9	run	16384	1	0	343	schedbench
10	run	16384	1	0	396	schedbench
...
-- 8 active process(es) --
=== xvtop refresh 2/3 ===
8	run	16384	10	0	3644	schedbench
9	run	16384	7	0	2517	schedbench
11	runble	16384	7	0	2640	schedbench
10	runble	16384	6	0	2177	schedbench
...
=== xvtop refresh 3/3 ===
8	run	16384	15	0	5498	schedbench
9	runble	16384	13	0	4741	schedbench
11	runble	16384	11	0	4162	schedbench
10	run	16384	9	0	3184	schedbench
...
```

This is the scheduling story made directly visible, not inferred: pid
6 is `schedbench`'s own parent, `sleep`ing in `wait()` for its four
children (8, 9, 10, 11 -- tickets 60, 20, 10, 5 respectively, matched
against their own final report below); already by refresh 1 every
child has been forked and is competing (`schedbench` forks its four
children back to back before any of them can finish, so all are
visible from the very first sample here -- unlike a slower host/QEMU
run where an earlier capture of this same session caught the
last-forked child still at 0 syscalls). By refresh 2 and 3, every
child's `RUNTICKS` and `SYSCALLS` are visibly climbing between
samples, live proof each was genuinely selected and ran real ticks in
between -- not merely `RUNNABLE`-but-starved. State alternates between
`run` and `runble` across refreshes as the lottery draw and the 3-hart
scheduler actually rotate which processes are executing at the instant
each sample was taken.

Once `schedbench` finishes (its own parent-collected, pipe-drained
report, printed only after `wait()`-ing for all four children --
Stage 3's own module comment in `user/schedbench.c` explains why this
pattern exists):

```
schedbench: report pid=8 tickets=60 start=5 end=35 elapsed=30 runticks=28 waitticks=0 selections=29
schedbench: report pid=9 tickets=20 start=5 end=35 elapsed=30 runticks=25 waitticks=0 selections=26
schedbench: report pid=10 tickets=10 start=5 end=35 elapsed=30 runticks=19 waitticks=0 selections=20
schedbench: report pid=11 tickets=5 start=5 end=35 elapsed=30 runticks=15 waitticks=0 selections=16
schedbench: done
```

Ticket order (60 > 20 > 10 > 5) is preserved exactly in both
`runticks` (28 > 25 > 19 > 15) and `selections` (29 > 26 > 20 > 16) --
the higher-ticket process was selected more and ran more, as
`SCHED_LOTTERY`'s weighted draw is supposed to produce. The *spread*
is visibly narrower than the 12:1 ticket ratio (60:5) -- `runticks`
here span under 2:1 (28:15), not 12:1 -- which is expected, not a
fairness bug: `TARGET_TICKS=30` is a fixed wall-clock competition
window shared by every run regardless of ticket count
(`user/schedbench.c`'s own module comment; also documented in
`docs/scheduler.md`), so every child accumulates real ticks within
that same window even at low ticket weight -- the ticket ratio governs
each draw's *probability*, not a hard proportional cutoff, and a
30-ish-tick window with this few total scheduling decisions carries
real sampling variance on top of the expected trend (an earlier
capture of this same session, discarded in favor of this one only
because a documentation-script fix landed in between, measured a
similarly-ordered but numerically different 30:25:22:11 spread --
consistent with this being real run-to-run PRNG/scheduling variance
around the same trend, not a specific number to treat as load-bearing).
See `benchmarks/raw/scheduler/` and `docs/scheduler.md` for dedicated,
larger-sample fairness benchmarks; this report's run is one live,
representative instance, not a fairness measurement in its own right.

## Stage 4 -- memory: lazy allocation and page-fault telemetry

```
$ vmfaulttest
vmfaulttest: start: pid=12 pagefaults=0 pagefaults_failed=0 sz=16384
vmfaulttest: after_sbrklazy: pid=12 pagefaults=0 pagefaults_failed=0 sz=32768
vmfaulttest: after_touch: pid=12 pagefaults=4 pagefaults_failed=0 sz=32768
vmfaulttest: after_retouch: pid=12 pagefaults=4 pagefaults_failed=0 sz=32768
vmfaulttest: done
```

Four stages, each an `xvstat(2)` snapshot of the *same* process (pid
12) at a different point: `sbrklazy(4 pages)` grows `sz` from 16384 to
32768 bytes immediately (`start` -> `after_sbrklazy`) with
**zero** new page faults -- growing the process's address space does
not itself touch memory, the defining property of the lazy-allocation
extension this project instruments (Phase 5, FR6, ADR-0011). Touching
one byte in each of the 4 pages (`after_touch`) produces exactly 4
counted faults, one per distinct page actually faulted in through
`vmfault()` (`kernel/vm.c`) -- not 1 (touching isn't coalesced) and not
more than 4 (each page faults exactly once). Touching the *same*
already-resident byte again (`after_retouch`) produces **no** further
faults -- `vmfault()`'s `ismapped()` check recognizes the page is
already backed and does nothing, exactly the invariant
`tests/vm/test_pagefault_counting.py` already asserts, now shown as a
live before/after snapshot rather than a pass/fail assertion.
`pagefaults_failed` stays at 0 throughout: nothing here comes close to
this build's ~128MB physical budget (see
[`docs/vm-extension.md`](vm-extension.md) for the deliberate-failure
counterpart to this trace, and [`docs/stress-testing.md`](stress-testing.md)
for the same fault path under genuine concurrent multi-hart pressure).

## Stage 5 -- exit and cleanup: what's left after the workload

```
$ xvtop 1 5
=== xvtop refresh 1/1 ===
PID	STATE	SZ	RUNTICKS	WAITTICKS	SYSCALLS	NAME
1	sleep	16384	0	34	24	init
2	sleep	20480	0	49	115	sh
13	run	20480	0	0	31	xvtop
-- 3 active process(es) --
```

Back down to exactly three processes -- `init`, `sh`, and this
invocation of `xvtop` itself (a fresh pid, 13, since every earlier
`xvtop`/`xtrace`/`schedbench`/`vmfaulttest` process has already exited
and been reaped by `sh`). Every `schedbench` child (pids 6, 8-11) and
`vmfaulttest` (pid 12) is gone -- not listed as `zombie`, not lingering
as stale telemetry -- direct, live confirmation of invariant #5
(zombie/free process slots never remain visible as active telemetry):
`xvtop`'s own zombie/`UNUSED` filter (Phase 3) plus `freeproc()`
resetting every telemetry field on free (Phases 1-5) together mean a
reaped process simply disappears from the next snapshot, cleanly.
`init`'s and `sh`'s own `WAITTICKS` (34, 49) climbed from 0 across the
whole session -- both spent the entire workload blocked in `wait()`/
waiting for shell input, exactly what their role predicts.

## Regression check (invariant #8)

```
$ echo observability_regression_ok
observability_regression_ok
```

The kernel is still fully healthy after tracing a real command,
running a live scheduler competition sampled through `xvtop`, and
driving the lazy-allocation fault path -- the same "still works
afterward" check every Phase 1-6 test in this repo ends with.

## What this report is not

Not a fairness benchmark (see `docs/scheduler.md`/
`benchmarks/methodology.md` for that, with a larger sample and a
head-to-head baseline-vs-lottery comparison), not a stress/correctness
proof under adversarial concurrent load (see `docs/stress-testing.md`
for that), and not new kernel or userspace code -- Phase 7 is
documentation over the instrumentation Phases 1-5 already built and
Phase 6 already stress-hardened, using exactly the tools (`xtrace`,
`xvtop`, `xvstat(2)`) an end user of this kernel would actually reach
for.
