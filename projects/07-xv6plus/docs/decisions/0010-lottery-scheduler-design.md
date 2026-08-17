# ADR-0010: Lottery scheduler design -- policy switch, PRNG, tickets lifecycle, zero-ticket floor

**Status:** Accepted; implemented in Phase 4
**Decision:** refines D4 (spec Â§1.10) -- the specific implementation
questions ADR-0004 deferred until the feature was actually built
**Date:** 2026-08-17

## Context

ADR-0004 committed to lottery scheduling as the Phase 4 scheduler
experiment but deferred every implementation question ("Not
implemented yet; recorded now... so later phases build toward one
settled direction"). Building it raised four concrete questions this
ADR settles, the same way ADR-0009 settled the specific locking
questions ADR-0006 deferred for Phase 2.

## Decisions

**1. Policy is a runtime switch, not a replacement.** `kernel/proc.c`
keeps the exact upstream round-robin loop, factored unchanged into
`schedule_roundrobin()`, and adds `schedule_lottery()` alongside it.
`scheduler()` dispatches between them per-call based on a global
`sched_policy` (`kernel/sched.h`: `SCHED_RR` / `SCHED_LOTTERY`),
defaulting to `SCHED_RR` so every Phase 0-3 test and upstream's own
`usertests` keep exercising unmodified baseline behavior unless a
program explicitly opts into `SCHED_LOTTERY` via the new
`schedpolicy(2)` syscall. This directly satisfies FR5's "policy
selection" requirement and keeps the change reviewable as a diff
against a known-good baseline rather than a rewrite.

**2. PRNG: fixed-seed xorshift32, not a hardware/entropy source.**
`lottery_rand()` is a 3-shift xorshift32 generator seeded once at boot
(`procinit()`) from a fixed constant (`SCHED_RNG_SEED`), not from
`ticks` or any other boot-time entropy. ADR-0004 required "a PRNG
seeded in a documented, reproducible way so any measured fairness
result can be reproduced exactly" -- xv6 has no hardware RNG wired up,
and a fixed seed is what makes a captured benchmark run
(`docs/scheduler.md`) a claim about *this specific, re-runnable*
sequence of draws, not a one-off. The PRNG state (`rng_state`) is
protected by a new, deliberately leaf lock, `sched_lock`, alongside
`sched_policy` -- never held while any `p->lock` is held, and nothing
reachable while holding it acquires a `p->lock`, so it cannot
introduce a lock-order cycle with the existing `wait_lock -> p->lock`
chain (invariant #2).

**3. Two-pass draw with a bounded retry, not a single locked pass.**
A weighted draw needs a ticket *total* before it can pick a winner,
but summing every `RUNNABLE` process's tickets and then walking the
table again to find the winner are two separate passes, each only
briefly holding one process's own `p->lock` at a time (never two at
once, preserving the existing "no path acquires two different
processes' `p->lock` at once" property ADR-0009 relies on). A
process's state can change between the two passes (it exits, blocks,
or another hart's draw picks it first). Two alternatives were
rejected: (a) holding every process's `p->lock` at once for the whole
draw -- a much larger, harder-to-reason-about lock-order surface for a
single teaching-kernel feature; (b) a single new giant lock
serializing the whole proc table against every other `p->lock` use --
defeats the purpose of per-process locking everywhere else in this
codebase. Instead, `schedule_lottery()` retries the whole two-pass
draw up to `SCHED_LOTTERY_MAX_ATTEMPTS` (4) times against fresh state,
and falls back to `pick_first_runnable()` (first-`RUNNABLE`-in-table-
order, i.e. one round-robin step) if every attempt misses. This keeps
invariant #4 ("the scheduler always eventually selects eligible work")
true by construction: the fallback cannot itself fail to find work
whenever `RUNNABLE` work actually exists.

**4. Tickets are inherited across `fork()`; selections are not.**
`p->tickets` follows Phase 1's `trace_mask` precedent (copied
parent-to-child in `kfork()`) rather than Phase 2's accounting-counter
precedent (reset to 0): a forked workload keeps its parent's
configured scheduling share until it calls `settickets()` itself,
which is the useful default for the "unequal priorities/tickets"
benchmark workloads spec Â§1.9 calls for (a process that `fork()`s
several workers expects them to keep its ticket weight, not restart at
the default). `p->sched_selections`, by contrast, follows the Phase 2
counter precedent (reset to 0, not inherited) -- a child's own
selection history starts at its own birth. Unlike `trace_mask`,
`settickets()`'s write to `p->tickets` *does* take `p->lock`
(ADR-0006 case (b), not case (a)): tickets is a real scheduling input
read cross-process by `schedule_lottery()` on any hart, not just a
monitoring display.

**5. Zero tickets are legal and deliberately not equivalent to "never
runs" -- but this is a narrower guarantee than it may first appear
(corrected during review followup).** `settickets(0)` is accepted, not
rejected -- a process can be configured to only ever run via
`pick_first_runnable()`'s fallback. That fallback triggers only when
`runnable_ticket_total()` returns exactly 0, i.e. only when *every*
currently-`RUNNABLE` process (not just the 0-ticket one) has 0
tickets, or there is no `RUNNABLE` process at all. `draw_and_run()`'s
weighted draw itself gives a 0-ticket process's cumulative range zero
width, so it can never win a weighted draw while any nonzero-ticket
process is also `RUNNABLE` -- that is correct, intended
proportional-share behavior (0 tickets = 0 share), not a bug. The
practical consequence: if at least one nonzero-ticket process stays
continuously `RUNNABLE`, a 0-ticket process competing against it is
never selected, for as long as that holds -- this is a real,
un-bounded-in-general limitation, not just a "rare" case. What IS true
unconditionally: the fallback keeps the scheduler itself from ever
stalling on `RUNNABLE` work (invariant #4's actual wording, "the
scheduler always eventually selects eligible work"), and in any
workload where the competing nonzero-ticket processes are themselves
finite (the common case for a benchmark run), the ticket total
eventually does hit 0 and every 0-ticket process gets picked before
the run ends -- exactly what
`tests/scheduler/test_lottery_zero_ticket_floor.py` and the captured
real run in `docs/scheduler.md` demonstrate (0-ticket children got
roughly 1/18th the CPU share of their 30-ticket peers, but every one
of them still completed). A hard, general starvation bound (e.g. a "compensation tickets" aging
mechanism) is not implemented here; see `docs/scheduler.md`'s
"Zero-ticket floor" section for the full, corrected writeup. (A
separate, unrelated correctness bound was added to the ticket count
itself during the same review followup -- see
[ADR-0013](0013-ticket-count-bound.md) -- but it does not change this
decision's answer.)

## Rationale

Every one of these five choices is the same move: reuse the existing,
already-reviewed locking and lifecycle conventions this codebase
established in Phases 1-2 (`p->lock` per case (a)/(b), the
owner-mutates/cross-process-read split, the fork-inherits-vs-resets
split) rather than inventing new ones for the scheduler specifically.
The one genuinely new lock (`sched_lock`) is scoped as narrowly as
possible (two small globals, never nested under a `p->lock`) for the
same reason ADR-0009 rejected reusing `tickslock` for `waitticks`: a
new lock-order edge is exactly the kind of thing invariant #2 exists
to catch before it ships, not after.

## Consequences

A future scheduling policy (e.g. a stretch-goal multilevel feedback
queue) has a template to follow: add a `schedule_*()` function beside
the existing two, add a policy constant to `kernel/sched.h`, and
decide its own answer to the same five questions above rather than
assuming lottery scheduling's answers apply. `SCHED_LOTTERY_MAX_ATTEMPTS`
is a correctness bound, not a tuned performance constant -- raising it
only matters if a future change makes the race between the two draw
passes measurably more frequent, which nothing in this codebase
currently does.
