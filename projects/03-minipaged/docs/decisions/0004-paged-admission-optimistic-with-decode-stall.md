# 0004 — Paged admission is optimistic (current need, not worst case); decode-time KV exhaustion stalls and retries

- **Status:** Accepted (Phase 2 scope; the deadlock risk this introduces
  is deliberately left open here and closed in Decision 0005 / Phase 3)
- **Phase:** 2
- **Related spec sections:** S1.4 core invariant #5, S1.9 ("Failure: KV
  exhaustion..." test-plan item), S1.10 D7 (preemption: reject at
  admission for MVP), FR4, FR5
- **Related code:** `kv/manager.py` (`PagedKVManager`, `PagedKVEngine`),
  `simulation/engine.py` (`_can_decode` hook,
  `EventType.DECODE_STALLED`), `tests/unit/test_manager.py`

## Context

Phase 1's `ContiguousKVEngine` admits a request only after reserving its
entire worst-case footprint (`prompt_len + max_new_tokens`) up front —
safe (a running sequence's `update_used` can never fail), but exactly the
"reserve one contiguous span sized to the worst case" waste Phase 2
exists to eliminate (see `docs/kv-memory.md`, Decision 0002).

Phase 2's whole point is that a paged sequence should not have to commit
to its worst case: it should hold only as many blocks as it has actually
generated, rounded up (Decision 0003). That means admission has a real
choice to make, and the two options have opposite failure modes:

- **Reserve the worst case at admission anyway** (in blocks, not
  tokens): safe — a decode step can never fail for lack of space — but
  then admission behaves identically to Phase 1's, and Phase 2 would
  demonstrate nothing paging-specific about *admission* (only about
  *physical layout*, which is a real but smaller claim).
- **Admit based on current need only** (`prompt_len`, rounded up):
  demonstrates paging's actual benefit — `examples/phase2_paged_kv_demo.py`
  shows peak waste ratio drop from 63.7% (contiguous) to 31.5% (paged) on
  an identical trace and capacity — but a decode step can now genuinely
  fail: the pool may have no free block left for a sequence's next token,
  even though every currently-running sequence was legitimately admitted.

## Decision

`PagedKVEngine`/`PagedKVManager.admit` take the second option: admission
needs only `blocks_needed_for(prompt_len)` free blocks, never
`prompt_len + max_new_tokens`. Two admission outcomes, deliberately
mirroring `ContiguousKVEngine`'s (see `docs/decisions/0002-...md`) so the
two engines are directly comparable on the same trace:

- **Hard cancel** (D7 default: reject at admission) if the request's own
  worst case (`blocks_needed_for(prompt_len + max_new_tokens)`) exceeds
  *total* pool capacity — no amount of waiting fixes that, so it is
  rejected immediately (`ADMISSION_REJECTED` + `CANCELLED`), never left
  queued forever.
- **Soft reject** (stays queued, retried next tick) if currently-free
  blocks are insufficient for `prompt_len` alone right now, even though
  the request's own worst case *would* fit total capacity eventually.

Decode-time growth is gated by a new engine hook,
`_can_decode(seq)` → `PagedKVManager.can_grow_to`: a pure, non-mutating
check run immediately before `step_decode()`. If it returns False, the
sequence is left exactly as it was (`tokens_generated` unchanged, no
partial/corrupt growth) and an `EventType.DECODE_STALLED` event is
recorded instead of `DECODE_STEP` — the sequence is retried automatically
next tick, since every tick re-attempts every running sequence
unconditionally. This is what turns "KV exhaustion mid-decode" (an S1.9
failure-mode test-plan item) into a safe, observable, testable outcome
instead of a crash, a silently-dropped token, or corrupted block-table
state.

## Consequences: this can deadlock, and that is documented, not hidden

Optimistic admission with no worst-case reservation and no preemption (D7
default for MVP) has a real, constructible failure mode: two or more
sequences can each be legitimately admitted (their combined admission-time
need fits, and each one's *individual* worst case fits total capacity
alone), then all simultaneously need one more block on the same tick with
none free. None can progress, none completes, none releases — permanent
deadlock, not a transient stall.
`tests/unit/test_manager.py::test_two_sequences_can_deadlock_under_optimistic_admission`
constructs exactly this (3 one-token blocks total, two sequences each
needing up to 3 tokens) and asserts `run_to_completion` raises
`RuntimeError` via its `max_ticks` safety net rather than hanging.

This is a **known, deliberate Phase 2 limitation**, not a bug to be
silently worked around:

- Per the roadmap's strict dependency order, Phase 2 delivers the paged
  *allocator* (FR5); admission control that provably prevents this
  scenario is explicitly Phase 3 scope — `docs/scheduler.md` already
  previewed this split before Phase 2 landed: "Admission control tied to
  real KV capacity (Phase 2's paged allocator, not Phase 1's contiguous
  stand-in) — core invariant #5 ... becomes a scheduler-level test, not
  just an allocator-level one."
- Decision 0005 (Phase 3) adds a block-quantized worst-case admission
  ledger on top of this same `PagedKVManager`, unchanged, which makes
  this exact deadlock scenario provably unreachable (proven by a property
  test, not just documented) — while still keeping physical allocation
  lazy, so paging's memory-efficiency benefit is not lost.
- Demo and non-adversarial test traces in this codebase
  (`examples/phase2_paged_kv_demo.py`,
  `tests/unit/test_manager.py::test_variable_length_trace_produces_utilization_statistics`)
  use capacity comfortably larger than the trace's concurrent demand, so
  this failure mode does not occur in them — deliberately, and noted here
  so nothing about Phase 2's "same trace runs paged and produces
  utilization statistics" exit criterion overstates what Phase 2 alone
  guarantees.
