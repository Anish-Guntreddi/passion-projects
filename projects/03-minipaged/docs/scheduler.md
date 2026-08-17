# Scheduler

Status: **not implemented yet.** This document exists now (rather than
being added when Phase 3 lands) because the spec's repository structure
requires `docs/scheduler.md` up front and because Phase 1's admission
policy already makes a couple of scheduler-adjacent decisions that are
worth recording before they are forgotten.

## What exists today (Phase 0-1): admission, not scheduling

`SimulationEngine._try_admit` / `ContiguousKVEngine._try_admit`
(`src/minipaged/kv/contiguous.py`) implement **FCFS admission**, not a
scheduler:

- Every tick, arrived-but-not-yet-admitted requests are retried oldest
  arrival first.
- Phase 0's base admission never fails (no capacity model).
- Phase 1's admission fails "soft" (stays queued, retried next tick) when
  free capacity is temporarily insufficient, or "hard"
  (`CapacityError` -> immediate `CANCELLED`) when the request's worst-case
  reservation exceeds *total* capacity and could never be admitted no
  matter how long it waits.
- There is no batching, no token/sequence budget, and no
  prefill-vs-decode work selection — every running sequence gets exactly
  one decode step per tick, unconditionally.

This deliberately simple policy is *not* `minipaged.scheduler` and will
not be reused unchanged once Phase 3 lands (see
`src/minipaged/scheduler/__init__.py`'s docstring).

## What Phase 3 adds (per spec FR4, FR9, D3-D5)

- **Continuous batching** with a per-step token or sequence budget (D3
  default: FCFS + per-step token budget) instead of "decode everything
  running, unconditionally."
- **`SchedulerDecision`** (FR9): the public interface describing one
  scheduling step's selected requests, prefill vs. decode work, token
  budget, expected KV growth, and any preemptions/rejections.
- **Prefill prioritization vs. decode latency** (D4 default: decode-first
  with a bounded prefill chunk per step) — prefill stops being "free and
  instantaneous" (Phase 0-1's documented simplification) and becomes real
  scheduled work competing with decode for the same per-step budget.
- **A fairness policy**, documented explicitly (D5 default: FCFS, with a
  starvation note) rather than left implicit.
- **Admission control tied to real KV capacity** (Phase 2's paged
  allocator, not Phase 1's contiguous stand-in) — core invariant #5
  ("scheduler never admits work requiring more KV capacity than policy
  permits") becomes a scheduler-level test, not just an allocator-level
  one.

## Exit criterion (spec Part 3)

"Runtime replays arrivals and produces scheduling timelines." This
document will be rewritten (not just appended to) once that lands.
