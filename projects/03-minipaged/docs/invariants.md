# Core invariants

From spec S1.4: "each becomes an explicit test." This table tracks, per
invariant, whether it applies yet (some are Phase 2+ concepts — physical
blocks, refcounts — that do not exist until the paged allocator lands)
and exactly where it is tested today.

| # | Invariant | Applies as of | Tested by |
|---|-----------|----------------|-----------|
| 1 | No physical block is simultaneously free and referenced. | Phase 2 (paged blocks don't exist yet) | — (Phase 1's `ContiguousMemoryModel` has an analogous property: a reservation cannot be both released and holding `used > 0`, enforced by `release()` popping the reservation entirely — see `test_contiguous.py::test_release_frees_the_reservation_slots`) |
| 2 | `ref_count` equals the number of active logical references where sharing is enabled. | Phase 4 (prefix sharing) | — |
| 3 | A sequence's logical block count covers its KV length. | Phase 2 (block tables don't exist yet) | Phase 1's analogous check: `used <= reserved` always, enforced by `ContiguousMemoryModel.update_used` raising if `used > res.reserved` — `test_contiguous.py::test_update_used_rejects_exceeding_reservation` |
| 4 | Releasing a sequence eventually returns all unshared blocks. | Phase 1 (contiguous reservations) onward | `test_contiguous.py::test_release_frees_the_reservation_slots`; `test_engine.py::test_completed_sequences_are_removed_from_running` (base lifecycle) |
| 5 | Scheduler never admits work requiring more KV capacity than policy permits. | Phase 1 (admission, not yet a real scheduler) onward | `test_contiguous.py::test_admission_is_rejected_when_capacity_is_insufficient`, `test_contiguous.py::test_request_larger_than_total_capacity_raises_capacity_error` |
| 6 | Completed/cancelled requests cannot remain scheduled. | Phase 0 (`Request` state machine) onward | `test_request.py::test_illegal_transitions_raise_and_do_not_mutate_state`, `test_request.py::test_completed_requests_cannot_be_rescheduled`, `test_request.py::test_cancelled_requests_cannot_be_rescheduled` |
| 7 | Deterministic simulated traces reproduce identical allocation history under fixed seed/config. | Phase 0 onward | `test_trace.py::test_generate_trace_is_deterministic_given_seed` (trace generation only); `tests/property/test_determinism.py` (hypothesis property: two independent engines replaying the same trace produce identical event logs); `tests/integration/test_trace_replay.py` (end-to-end replay determinism, including the `ContiguousKVEngine`) |

## Notes on invariants 1-3 before Phase 2

Invariants 1-3 are stated in terms of *physical/logical blocks*, which
are a Phase 2 concept (`kv/block.py`, `kv/table.py` — not yet
implemented). Phase 1's `ContiguousMemoryModel` has no blocks at all — one
reservation is one contiguous span — so there is no block-level free/
referenced distinction to violate yet. The table above lists the closest
Phase 1 analogue for each so the invariant isn't silently unaccounted-for,
but the *literal* block-based tests land with Phase 2 and this document
will be updated then, not before.

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
