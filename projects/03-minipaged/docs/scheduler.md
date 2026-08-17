# Scheduler

Status: **implemented (Phase 3)**. `SchedulerEngine`
(`src/minipaged/scheduler/scheduler.py`) subclasses Phase 2's
`PagedKVEngine` unchanged and adds continuous batching with a per-step
token budget plus a KV-capacity-aware admission ledger. See
`docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md` for
the full design rationale (D3, D4, D5) and
`docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md` for
the Phase 2 limitation this phase closes.

## What Phase 0-2 were: admission, not scheduling

`SimulationEngine._try_admit` (Phase 0, unconditional),
`ContiguousKVEngine._try_admit` (Phase 1, contiguous worst-case
reservation), and `PagedKVEngine._try_admit` (Phase 2, paged current-need
admission) are each a **simple, single-allocator admission policy** —
FCFS-ordered, retried every tick, but with no per-step work budget and no
notion of "how much concurrent work should be running at once beyond what
this one allocator's capacity check allows." Every running sequence gets
exactly one decode step per tick, unconditionally (Phase 2's `_can_decode`
aside, which only ever says "not yet," never "not this many"). None of
these three classes are replaced by `SchedulerEngine` — they remain the
right tool for isolating one allocator's behavior in tests
(`tests/unit/test_contiguous.py`, `tests/unit/test_manager.py`), and
`SchedulerEngine` inherits `PagedKVEngine`'s block-allocation machinery
rather than duplicating it.

## What `SchedulerEngine` adds (FR4, FR9, D3-D5)

- **Continuous batching with a per-step token budget** (D3 default):
  `token_budget` tokens of new admission (prefill) work per tick, FCFS-
  ordered, strictly bounded — decode work is never throttled by it (see
  next bullet). Batch composition changes tick to tick as budget and KV
  capacity allow, not admitted all at once
  (`tests/unit/test_scheduler.py::test_batch_composition_changes_across_ticks_continuous_batching`,
  `tests/integration/test_scheduler_replay.py::test_continuous_batching_is_observable_in_the_timeline`).
- **Decode-first prioritization** (D4 default): every running,
  non-stalled sequence gets its decode step every tick unconditionally —
  the token budget governs only new admission. "Bounded prefill chunk per
  step" means *how many requests* get admitted this tick, not a single
  request's prefill split across multiple ticks (that finer-grained
  chunked prefill is a stretch goal, not implemented).
- **`SchedulerDecision`** (FR9, `scheduler/decision.py`): the public
  interface for one scheduling step's outcome — admitted/rejected/
  cancelled/decoded/stalled/completed request ids, prefill tokens used
  vs. budget, post-tick KV block accounting, and `expected_kv_growth_blocks`
  (FR9's "expected KV growth": the admission ledger's total worst-case
  watermark reservation minus what's physically allocated right now — how
  many more blocks are already committed, in the worst case, before any
  currently running sequence can complete). Accumulated in
  `SchedulerEngine.decisions`, one per `step()` call — the "scheduling
  timeline" that is this phase's actual exit-criterion deliverable.
- **Admission control tied to real (paged) KV capacity, closing Phase
  2's deadlock gap**: a block-quantized worst-case admission ledger
  (reusing `ContiguousMemoryModel` at block granularity, purely as a gate
  — physical allocation stays lazy via the inherited `PagedKVManager`).
  This makes core invariant #5 hold at the scheduler level with a proof,
  not just an allocator-level check:
  `tests/property/test_scheduler_invariants.py` shows `DECODE_STALLED`
  never occurs across 100 randomized traces, and
  `tests/unit/test_scheduler.py::test_the_phase2_deadlock_scenario_completes_successfully_here`
  replays Decision 0004's exact hand-built deadlock scenario and confirms
  it now completes.
- **A documented fairness policy** (D5 default: FCFS): admission attempts
  run oldest-arrival-first every tick, and the first candidate that
  cannot currently be satisfied (soft rejection — insufficient ledger
  capacity or prefill budget *right now*) stops the attempt loop for the
  rest of that tick, so a later, smaller request never jumps the queue
  ahead of an earlier, larger one. **Starvation note**: this trades
  throughput for a hard fairness guarantee — a sustained stream of
  arrivals whose combined admission-time need chronically exceeds
  available budget/capacity can, in principle, delay an early large
  request indefinitely behind ongoing admission pressure from work that
  arrived after it, but it can never be *skipped past* by later, smaller
  work in the same tick. See
  `docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md`
  for the full argument and the bin-packing alternative it was chosen
  over. A genuinely unrecoverable request (worst case exceeds total KV
  capacity, or its prompt alone exceeds the whole per-tick token budget)
  is still cancelled immediately, per D7, and never blocks the queue.

## Exit criterion (spec Part 3)

"Runtime replays arrivals and produces scheduling timelines." Demonstrated
by `examples/phase3_scheduler_demo.py` (prints the full per-tick
timeline: admitted/decoded/rejected/cancelled/completed counts, prefill
budget usage, KV block accounting) and asserted end to end by
`tests/integration/test_scheduler_replay.py`.

## What is deliberately still not here

No preemption, no recompute, no swap (D7 default, unchanged since
Decision 0002) — once genuinely admitted, the admission ledger guarantees
a sequence can always reach completion without ever being evicted.
Chunked prefill (splitting one request's prefill across multiple ticks)
remains a roadmap stretch goal. Prefix sharing / copy-on-write is Phase
4 scope and does not touch the scheduler.
