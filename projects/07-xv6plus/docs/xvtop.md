# xvtop (FR4, Phase 3)

## What it is

A userspace process monitor, `user/xvtop.c`, built entirely on top of
the Phase 2 `xvstat(2)` interface (`docs/accounting.md`). It has no
special kernel access of its own: it is index-based, not pid-based --
exactly the shape a real process monitor's problem takes, since
`ps`/`top` don't get to ask the kernel "show me PID 17," they
enumerate the process table and filter (see `user/acct.h`'s comment).

## Usage

```
xvtop [count [period]]
  count  -- number of refreshes to print (default 1)
  period -- ticks to pause() between refreshes (default 10, roughly
            one second at this kernel's ~10 Hz timer)
```

## Behavior

Each refresh:

1. Enumerates every proc-table slot via `xvstat(2)`, `idx = 0` upward,
   stopping as soon as `xvstat` reports `idx` out of range (i.e. it
   has reached `NPROC`).
2. Filters out `UNUSED` **and** `ZOMBIE` slots (invariant #5: neither
   a free slot nor an exited-but-unreaped one is an active process --
   `ZOMBIE` is a real, non-`UNUSED` state, so it needs its own explicit
   check, not just "skip the zero pid" or "skip `UNUSED`" alone).
3. Sorts what's left by `runticks` descending (busiest process first)
   -- the "sorting ... if manageable" half of FR4, implemented as a
   plain insertion sort since `NPROC` (64) is small enough that
   anything fancier would be optimizing before it's needed.
4. Prints one line per surviving row: pid, state, memory size (bytes),
   runticks, waitticks, syscalls, name; then a trailing
   `-- N active process(es) --` summary line, where `N` is the number
   of rows actually printed (i.e. already post-filter).

## Example transcript

Captured from a real `scripts/run-tests.sh` session
(`xvtop 2 5`, run from an interactive shell with `init` and `sh` also
alive):

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

Reading this: `xvtop` itself always appears in its own listing (state
`run`, since it is the process currently executing when it takes the
snapshot); `init` and `sh` are both `sleep`ing on I/O between shell
commands; between the two refreshes (5 ticks apart, per the `period`
argument) `xvtop`'s own `waitticks` and `syscalls` climb from its
`pause(5)` and its own `xvstat()` polling loop.

## Zombie filtering

Point 2 above is directly exercised by
`tests/xvtop/test_xvtop_zombie_filtered.py`, via a small dedicated
support program, `user/xvtopzombie.c`: it forks a child that exits
immediately and is deliberately never `wait()`ed on (a genuine,
reproducible zombie), polls `xvstat(2)` for that exact pid until it
observes `XV_ZOMBIE` (removing any timing race), then `exec()`s
straight into `xvtop`. Because `kexec()` replaces only the program
image, not the pid or its children, the zombie is still there for
`xvtop`'s first refresh -- and must not appear as a row.

## Known limitation: no single atomic whole-table snapshot

Each `xvstat(2)` call is its own point-in-time read of one slot, taken
under that slot's own `p->lock` (ADR-0009) -- there is no single lock
held across the whole `idx = 0..NPROC-1` scan. A process can change
state between the call for slot *i* and the call for slot *i+1*, so a
single `xvtop` refresh is a fast, best-effort composite of
per-process-consistent snapshots, not one instant-in-time snapshot of
the entire system. This is consistent with invariant #8 (observability
must not destabilize the kernel, not "observability is perfectly
atomic") and is the same tradeoff any real `ps`/`top` implementation
accepts.

## Test coverage

See `tests/xvtop/`:

- `test_xvtop_basic.py` -- Phase 3 exit criterion ("Tool visibly
  reports active processes and resource data"): running `xvtop`
  produces the expected header, includes `xvtop`'s own row, and prints
  a correctly-formatted summary line, across multiple refreshes.
- `test_xvtop_zombie_filtered.py` -- invariant #5: a deterministically
  constructed zombie process (`user/xvtopzombie.c`) never appears as a
  row.

Run via `scripts/run-tests.sh`.
