# ADR-0013: Ticket count upper bound (`SCHED_MAX_TICKETS`)

**Status:** Accepted; implemented as a review-followup fix on top of
Phase 4
**Decision:** a correctness fix to ADR-0010's `settickets(2)`
validation, not a new open decision from the spec
**Date:** 2026-08-17

## Context

An external review (Codex) found a real, reachable correctness bug in
the lottery draw: `draw_and_run()` (`kernel/proc.c`) computes
`winner = lottery_rand() % total`, where `lottery_rand()` is a
xorshift32 PRNG returning a plain `uint32`, and `total` is a `uint64`
sum of every `RUNNABLE` process's `p->tickets`
(`runnable_ticket_total()`). Before this fix, `sched_settickets(int n)`
accepted any `n >= 0` with no upper bound.

If the sum of `RUNNABLE` tickets ever exceeded `UINT32_MAX` (~4.29
billion) -- reachable with a modest number of large `settickets()`
calls, since it is a public, unrestricted syscall (same "no
process-identity check" design as `schedpolicy()`, ADR-0010 decision
1) -- `winner` could never fall in the cumulative-ticket range beyond
2^32. Any process whose cumulative range in `draw_and_run()`'s table
walk started past that point would become structurally unreachable by
any draw: not disadvantaged, but permanently excluded, a real
correctness bug distinct from the (already-documented, ADR-0010
decision 5) zero-ticket case.

## Decision

`kernel/sched.h` gains `SCHED_MAX_TICKETS` (100000).
`sched_settickets(int n)` (`kernel/proc.c`) now rejects
`n > SCHED_MAX_TICKETS` the same way it already rejects `n < 0`
(returns -1, changes nothing).

**Why 100000, not some other value.** `NPROC` (`kernel/param.h`) is
64. Even if every single process slot were simultaneously maxed out
at `SCHED_MAX_TICKETS`, the total (`64 * 100000 = 6,400,000`) is still
over 600x below `UINT32_MAX` (~4.29 billion) -- nowhere near the
truncation boundary this bound exists to avoid, with a wide safety
margin for headroom. At the same time, 100000 is five orders of
magnitude above the widest ticket spread any benchmark workload in
this project actually uses (`docs/scheduler.md`: 60 vs. 5, a 12x
spread), so the bound cannot realistically constrain any real
experiment here -- it only closes the unreachable-process failure
mode.

**Why a fixed constant, not a dynamic bound scaled to `NPROC` at
runtime.** `NPROC` is a compile-time constant in this codebase (as it
is in upstream xv6), so a static bound chosen against it is exactly as
safe as a computed one would be, without adding a runtime computation
to every `settickets()` call for no behavioral benefit.

## Consequences

`tixvalidate.c` and `tests/scheduler/test_settickets_validation.py`
now also exercise the new boundary (`SCHED_MAX_TICKETS` itself
accepted, `SCHED_MAX_TICKETS + 1` rejected), alongside the pre-existing
negative/zero/positive cases. No other behavior changes: every
existing benchmark and test in this project uses ticket values far
below this bound, so this fix is invisible to them.
