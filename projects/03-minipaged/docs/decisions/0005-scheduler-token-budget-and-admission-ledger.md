# 0005 — Scheduler: per-step token budget, decode-first prioritization, block-quantized admission ledger, strict FCFS (D3, D4, D5)

- **Status:** Accepted
- **Phase:** 3
- **Related spec sections:** S1.4 core invariant #5, S1.10 D3
  (scheduler policy + token budget → default: FCFS with per-step token
  budget), D4 (prefill vs. decode latency → default: decode-first with
  bounded prefill chunk per step), D5 (fairness policy → default: FCFS +
  starvation note), D7 (preemption: reject at admission for MVP — see
  Decision 0002), FR4, FR9
- **Related code:** `scheduler/scheduler.py` (`SchedulerEngine`),
  `scheduler/decision.py` (`SchedulerDecision`)

## Context

Phase 2's `PagedKVEngine` admits requests on current need alone (Decision
0004) — real, but with a genuine, constructed deadlock risk when nothing
bounds how much concurrently-running work the pool as a whole is
committed to. `docs/scheduler.md` previewed the fix before Phase 2 even
landed: "Admission control tied to real KV capacity ... core invariant
#5 ... becomes a scheduler-level test, not just an allocator-level one."
D3-D5 are three tightly coupled decisions about that same scheduler, so
one ADR covers all three rather than splitting a single design into
three documents.

## Decision

`SchedulerEngine` subclasses `PagedKVEngine` unchanged (same `BlockPool`
/ `BlockTable` / lazy physical allocation, same `_can_decode` stall hook)
and adds exactly two mechanisms:

**1. Per-step token budget, decode-first (D3, D4).** Each tick has a
configured `token_budget` (D3 default). Decode work is *never* throttled
by it — every running, non-stalled sequence gets its decode step every
tick, unconditionally, exactly as in every earlier phase (D4: "decode-
first" is read literally as "decode is never the thing that waits").
The budget instead bounds only new admission (prefill) per tick: each
newly admitted request's `prompt_len` is charged against the tick's
remaining budget, and once the next FCFS candidate's prompt cannot fit
what is left, no further admission is attempted this tick (D4's "bounded
prefill chunk per step" — the chunk being *which requests* get admitted
this tick, not a single request's prefill split across multiple ticks;
that finer-grained chunked prefill remains a stretch goal per the
roadmap, not implemented here). This is what makes batch composition
change dynamically tick to tick — continuous batching (FR4) — instead of
admitting everything the pool can physically hold in one shot.

**2. Block-quantized worst-case admission ledger (closes Decision
0004's gap).** `SchedulerEngine` reuses Phase 1's `ContiguousMemoryModel`
(`docs/decisions/0002-...md`) at *block* granularity — `capacity_tokens
= num_blocks`, not raw tokens — purely as an admission gate:
`try_reserve(request_id, size=watermark, used=prompt_blocks)` where
`watermark = blocks_needed_for(prompt_len + max_new_tokens)`. Physical
block allocation stays entirely with the inherited `PagedKVManager`,
lazy and block-granular exactly as in Phase 2 — the ledger never touches
a real `PhysicalBlock`. This split (conservative admission watermark vs.
lazy physical allocation) is the entire trick: it makes deadlock
structurally unreachable — `sum(watermark_i) <= num_blocks` for every
admitted sequence, and a sequence's actual physical usage is always
`<= its own watermark`, so `sum(physical_i) <= sum(watermark_i) <=
num_blocks` always — while `PagedKVMetrics`/utilization reporting still
reflects the tighter, lazy physical numbers, not the conservative
watermark, so paging's memory-efficiency benefit (Decision 0003's
measured 31.5% vs. 63.7% peak waste ratio) is not lost by this change.

`tests/property/test_scheduler_invariants.py` proves this claim directly
(100 hypothesis examples across varied pool sizes, block sizes, and token
budgets: every replay completes, `DECODE_STALLED` never appears in the
event log), and
`tests/unit/test_scheduler.py::test_the_phase2_deadlock_scenario_completes_successfully_here`
replays Decision 0004's exact hand-constructed deadlock trace through
`SchedulerEngine` and confirms it now completes.

**Fairness (D5): strict FCFS, documented starvation tradeoff.** Both
mechanisms are gated in oldest-arrival-first order every tick
(`_arrived_waiting_fcfs`, inherited from `SimulationEngine`). Crucially,
the *first* candidate this tick's budget or ledger cannot currently
satisfy **stops the admission attempt loop for the rest of this tick** —
a later, smaller request is never allowed to jump the queue ahead of an
earlier, larger one just because it happens to fit the leftover
budget/capacity
(`tests/unit/test_scheduler.py::test_soft_rejection_blocks_later_smaller_requests_in_the_same_tick`).
This is a deliberate choice over a greedier bin-packing policy that would
try every waiting request every tick (as Phase 1's `ContiguousKVEngine`
and Phase 2's `PagedKVEngine` still do, unchanged — this ADR does not
retroactively change their behavior): bin-packing has strictly higher
per-tick throughput/utilization, but can starve a large request
indefinitely if a stream of smaller ones keeps arriving and always fits
whatever budget/capacity the large one didn't. Strict FCFS trades some
throughput for a hard fairness guarantee: once a request is the oldest
waiting one, nothing arriving later can ever be served ahead of it. A
genuinely unrecoverable request (worst case exceeds total KV capacity, or
its prompt alone exceeds the *entire* per-tick token budget so it could
never fit even with a completely idle pool) is still cancelled
immediately and does **not** block the queue — same D7-aligned pattern as
every earlier phase.

## Consequences

- `SchedulerDecision` (`decision.py`, FR9) is appended once per `step()`
  call to `SchedulerEngine.decisions` — the "scheduling timeline" that is
  this phase's exit criterion ("Runtime replays arrivals and produces
  scheduling timelines"). Its `stalled` field exists for structural
  parity with FR9's "preemptions/rejections if implemented" but is
  proven always empty by this decision's admission ledger — not dead
  code, a documented, tested guarantee. Its `expected_kv_growth_blocks`
  field (FR9's "expected KV growth") is this same ledger's
  `reserved_tokens` minus what's physically allocated — the worst-case
  headroom this decision reads off directly rather than recomputing.
- Admission is strictly more conservative than Phase 2's `PagedKVEngine`
  on the same trace/capacity
  (`tests/unit/test_scheduler.py::test_ledger_reserves_worst_case_more_conservatively_than_paged_manager_alone`):
  a second request that Phase 2 would admit immediately can be soft-
  rejected here because the first request's ledger watermark already
  committed the pool's remaining capacity on paper, even though nothing
  is physically allocated yet. This is the deliberate cost of the
  deadlock-freedom guarantee — safety over maximum instantaneous
  concurrency, matching D7's overall MVP stance (reject/defer, not
  preempt or overcommit).
- No preemption, no recompute, no swap (D7 default, unchanged from
  Decision 0002): once genuinely admitted, a sequence's ledger
  reservation guarantees it can always reach completion; nothing ever
  evicts a running sequence to make room for another.
