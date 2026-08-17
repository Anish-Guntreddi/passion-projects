# Core invariants

From spec S1.4: "each becomes an explicit test." This table tracks, per
invariant, whether it applies yet (invariant #2 is a Phase 4 concept —
shared references — that does not exist until prefix sharing lands) and
exactly where it is tested today.

| # | Invariant | Applies as of | Tested by |
|---|-----------|----------------|-----------|
| 1 | No physical block is simultaneously free and referenced. | Phase 2 (literal blocks) | `test_pool.py::test_no_block_is_simultaneously_free_and_referenced`; `test_pool.py::test_free_rejects_a_block_with_more_than_one_referrer`/`test_release_rejects_double_release` (the operations that would violate it are refused outright); `tests/property/test_paged_kv_invariants.py` (random admit/grow/release sequences, `free + allocated == total` after every operation). Phase 1's `ContiguousMemoryModel` still has an analogous property pre-Phase-2 — see `test_contiguous.py::test_release_frees_the_reservation_slots`. |
| 2 | `ref_count` equals the number of active logical references where sharing is enabled. | Phase 4 (prefix sharing) | — (`BlockPool.retain`/`release` already implement the general refcount primitives per FR5 and are unit-tested at the pool level — `test_pool.py::test_retain_increments_ref_count`, `test_release_decrements_and_frees_at_zero` — but nothing calls `retain` yet, since no block has more than one owner until Phase 4) |
| 3 | A sequence's logical block count covers its KV length. | Phase 2 (literal blocks) | `test_table.py::test_a_sequences_logical_block_count_always_covers_its_kv_length`; `tests/property/test_paged_kv_invariants.py` (`capacity_tokens_for(request_id) >= length` after every grow). Phase 1's analogous check pre-Phase-2: `used <= reserved` always — `test_contiguous.py::test_update_used_rejects_exceeding_reservation`. |
| 4 | Releasing a sequence eventually returns all unshared blocks. | Phase 1 onward | Phase 1: `test_contiguous.py::test_release_frees_the_reservation_slots`. Phase 2: `test_table.py::test_free_all_returns_every_block_and_clears_the_table`, `test_manager.py::test_release_frees_every_block_the_request_held`, `tests/property/test_paged_kv_invariants.py` ("draining every remaining live request returns all capacity: no leaks"). Base lifecycle: `test_engine.py::test_completed_sequences_are_removed_from_running`. |
| 5 | Scheduler never admits work requiring more KV capacity than policy permits. | Phase 1 (admission) onward; airtight at the *scheduler* level from Phase 3 | Phase 1: `test_contiguous.py::test_admission_is_rejected_when_capacity_is_insufficient`, `test_request_larger_than_total_capacity_raises_capacity_error`. Phase 2 (weaker — see the caveat below): `test_manager.py::test_admission_is_soft_rejected_when_free_blocks_insufficient`, `test_request_whose_worst_case_exceeds_total_capacity_is_cancelled`. Phase 3 (proven, not just checked): `tests/property/test_scheduler_invariants.py::test_scheduler_admission_never_lets_decode_exhaust_kv_capacity` (100 randomized traces, `DECODE_STALLED` never occurs); `test_scheduler.py::test_the_phase2_deadlock_scenario_completes_successfully_here`. |
| 6 | Completed/cancelled requests cannot remain scheduled. | Phase 0 (`Request` state machine) onward | `test_request.py::test_illegal_transitions_raise_and_do_not_mutate_state`, `test_request.py::test_completed_requests_cannot_be_rescheduled`, `test_request.py::test_cancelled_requests_cannot_be_rescheduled` |
| 7 | Deterministic simulated traces reproduce identical allocation history under fixed seed/config. | Phase 0 onward | `test_trace.py::test_generate_trace_is_deterministic_given_seed` (trace generation only); `tests/property/test_determinism.py` (hypothesis property, one test per engine class — base, contiguous, paged, scheduler — replaying the same trace through two independent engines produces identical event logs, and for `SchedulerEngine`, identical `decisions` timelines too); `tests/integration/test_trace_replay.py` / `tests/integration/test_scheduler_replay.py` (end-to-end replay determinism) |

## Important caveat on invariant #5 at Phase 2 alone

`PagedKVEngine` (Phase 2, no scheduler) does **not** fully satisfy
invariant #5 by itself: it admits based on current need, not worst case,
so multiple legitimately-admitted sequences can jointly demand more
blocks than the pool holds and deadlock —
`test_manager.py::test_two_sequences_can_deadlock_under_optimistic_admission`
demonstrates this directly, and
`docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md`
documents it as a deliberate, known Phase 2 limitation rather than a bug.
Phase 3's `SchedulerEngine` (same allocator, block-quantized admission
ledger added on top) is what makes invariant #5 hold *provably*, not just
in hand-picked test cases — see
`docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md`.

## Why invariant 7 needed a dedicated fix

Request ids were originally drawn from a single module-level counter
shared by every `SimulationEngine` in the process
(`minipaged.requests.request.next_request_id`). Loading the *same* trace
twice in one process — exactly what a benchmark harness comparing
contiguous vs. paged allocation on identical arrivals would do — produced
different request ids, and therefore a different event log, on the
second load unless the caller manually called `reset_request_id_counter()`
first. `SimulationEngine` now owns its own `itertools.count`, so two
independent engines replaying the same trace in the same process are
identical by construction, with no caller-side reset required. See
`docs/decisions/0001-simulation-first-architecture.md` for the broader
determinism rationale and `tests/property/test_determinism.py` for the
regression test.
