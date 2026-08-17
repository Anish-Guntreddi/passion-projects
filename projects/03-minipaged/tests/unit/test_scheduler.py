from __future__ import annotations

import pytest

from minipaged.kv.manager import PagedKVEngine
from minipaged.requests.request import RequestState
from minipaged.scheduler.scheduler import SchedulerEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceRequestSpec


def _spec(prompt_len, max_new_tokens=1, target_output_len=None, arrival_time=0.0):
    return TraceRequestSpec(
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        target_output_len=target_output_len if target_output_len is not None else max_new_tokens,
        arrival_time=arrival_time,
    )


def test_rejects_non_positive_token_budget():
    with pytest.raises(ValueError, match="token_budget"):
        SchedulerEngine(num_blocks=10, token_budget=0)


# -- per-step prefill token budget: bounded admission chunk (D3/D4) ---------


def test_admission_this_tick_is_bounded_by_the_prefill_token_budget():
    engine = SchedulerEngine(num_blocks=100, block_size=1, token_budget=10, decode_tick=1.0)
    [req1, req2] = engine.load_trace(
        [_spec(prompt_len=6, arrival_time=0.0), _spec(prompt_len=6, arrival_time=0.0)]
    )
    engine.step()  # budget=10: req1 (6) fits, leaves 4 remaining -- req2 (6) does not
    assert req1.state == RequestState.RUNNING
    assert req2.state == RequestState.WAITING
    rejected = engine.event_log.for_request(req2.request_id)
    assert [e.event_type for e in rejected] == [EventType.ARRIVED, EventType.ADMISSION_REJECTED]
    assert rejected[1].detail["reason"] == "prefill_budget_exhausted_this_tick"

    engine.step()  # budget resets to 10 -- req2 now fits alone
    assert req2.state == RequestState.RUNNING


def test_batch_composition_changes_across_ticks_continuous_batching():
    """Not static/sequential batching: the running set is built up
    gradually as the per-step prefill budget allows, not all at once."""
    engine = SchedulerEngine(num_blocks=100, block_size=1, token_budget=5, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=5, arrival_time=0.0) for _ in range(4)])
    running_sizes = []
    for _ in range(4):
        engine.step()
        running_sizes.append(len(engine.running))
    assert running_sizes == [1, 1, 1, 1]  # one admitted per tick -- budget only fits one at a time
    assert len(engine.decisions) == 4
    assert [len(d.admitted) for d in engine.decisions] == [1, 1, 1, 1]


# -- strict FCFS: a soft rejection stops the tick's admission attempts -------


def test_soft_rejection_blocks_later_smaller_requests_in_the_same_tick():
    """D5 fairness note: once the oldest-arrival waiting request cannot be
    admitted this tick, later (even smaller, individually-fitting)
    requests are not tried either -- no queue-jumping."""
    engine = SchedulerEngine(num_blocks=100, block_size=1, token_budget=10, decode_tick=1.0)
    [req_a, req_b, req_c] = engine.load_trace(
        [
            _spec(prompt_len=8, arrival_time=0.0),  # admits, leaves 2 remaining
            _spec(prompt_len=5, arrival_time=0.0),  # needs 5, only 2 left -- soft reject, stop
            _spec(prompt_len=1, arrival_time=0.0),  # would fit (1 <= 2) but never attempted
        ]
    )
    engine.step()
    assert req_a.state == RequestState.RUNNING
    assert req_b.state == RequestState.WAITING
    assert req_c.state == RequestState.WAITING
    # req_c was never even attempted this tick: no ADMISSION_REJECTED event for it yet.
    assert engine.event_log.for_request(req_c.request_id) == [
        e
        for e in engine.event_log.for_request(req_c.request_id)
        if e.event_type == EventType.ARRIVED
    ]

    engine.step()  # fresh budget: req_b (5) then req_c (1) both fit
    assert req_b.state == RequestState.RUNNING
    assert req_c.state == RequestState.RUNNING


# -- hard cancellation: unrecoverable, never blocks the FCFS queue ----------


def test_request_whose_worst_case_exceeds_total_kv_capacity_is_cancelled():
    engine = SchedulerEngine(num_blocks=4, block_size=1, token_budget=100, decode_tick=1.0)
    [req] = engine.load_trace([_spec(prompt_len=1, max_new_tokens=10, arrival_time=0.0)])
    engine.step()
    assert req.state == RequestState.CANCELLED
    assert req in engine.cancelled
    types = [e.event_type for e in engine.event_log.for_request(req.request_id)]
    assert types == [EventType.ARRIVED, EventType.ADMISSION_REJECTED, EventType.CANCELLED]
    reject_event = engine.event_log.for_request(req.request_id)[1]
    assert reject_event.detail["reason"] == "exceeds_total_kv_capacity"


def test_request_whose_prompt_exceeds_the_step_token_budget_is_cancelled():
    engine = SchedulerEngine(num_blocks=1000, block_size=1, token_budget=5, decode_tick=1.0)
    [req] = engine.load_trace([_spec(prompt_len=6, max_new_tokens=1, arrival_time=0.0)])
    engine.step()
    assert req.state == RequestState.CANCELLED
    reject_event = engine.event_log.for_request(req.request_id)[1]
    assert reject_event.detail["reason"] == "prompt_exceeds_step_token_budget"


def test_hard_cancellation_does_not_block_a_later_admittable_request():
    engine = SchedulerEngine(num_blocks=1000, block_size=1, token_budget=100, decode_tick=1.0)
    [impossible, fine] = engine.load_trace(
        [
            _spec(prompt_len=1, max_new_tokens=10_000, arrival_time=0.0),  # worst case: way too big
            _spec(prompt_len=1, max_new_tokens=1, arrival_time=0.0),
        ]
    )
    engine.step()
    assert impossible.state == RequestState.CANCELLED
    assert fine.state == RequestState.RUNNING  # not blocked by the cancellation ahead of it


# -- admission ledger is a stricter gate than the paged manager alone -------


def test_ledger_reserves_worst_case_more_conservatively_than_paged_manager_alone():
    """Contrast with Phase 2's PagedKVEngine: the same second request that
    Phase 2 would admit immediately (physically, only 1 block is needed)
    is soft-rejected here, because the scheduler's ledger already
    committed the *first* request's entire worst-case footprint."""
    num_blocks, block_size = 4, 1
    specs = [
        _spec(prompt_len=1, max_new_tokens=3, arrival_time=0.0),  # worst case: 4 blocks
        _spec(prompt_len=1, max_new_tokens=3, arrival_time=0.0),
    ]

    paged = PagedKVEngine(num_blocks=num_blocks, block_size=block_size, decode_tick=1.0)
    [p1, p2] = paged.load_trace(specs)
    paged.step()
    assert p1.state == RequestState.RUNNING
    assert p2.state == RequestState.RUNNING  # Phase 2 admits both: only 1 block each needed now

    scheduled = SchedulerEngine(
        num_blocks=num_blocks, block_size=block_size, token_budget=100, decode_tick=1.0
    )
    [s1, s2] = scheduled.load_trace(specs)
    scheduled.step()
    assert s1.state == RequestState.RUNNING
    assert s2.state == RequestState.WAITING  # ledger already committed all 4 blocks to s1


def test_release_frees_both_the_ledger_and_the_physical_pool():
    engine = SchedulerEngine(num_blocks=10, block_size=1, token_budget=100, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=1, max_new_tokens=2, target_output_len=2)])
    engine.run_to_completion()
    assert engine.manager.pool.num_free == 10
    # The admission ledger is an internal detail, but its own conservation
    # property (everything reserved is eventually released) is worth a
    # direct check here rather than only inferring it from the pool.
    assert engine._ledger.reserved_tokens == 0


# -- SchedulerDecision: the scheduling timeline (FR9, exit criterion) -------


def test_step_records_a_scheduler_decision_with_expected_fields():
    engine = SchedulerEngine(num_blocks=100, block_size=1, token_budget=10, decode_tick=1.0)
    [req] = engine.load_trace([_spec(prompt_len=4, max_new_tokens=2, target_output_len=2)])
    engine.step()
    assert len(engine.decisions) == 1
    decision = engine.decisions[0]
    assert decision.tick == 1.0
    assert decision.admitted == (req.request_id,)
    assert decision.rejected == ()
    assert decision.cancelled == ()
    assert decision.decoded == ()  # just-admitted, skipped this tick's decode pass
    assert decision.stalled == ()
    assert decision.completed == ()
    assert decision.prefill_tokens_used == 4
    assert decision.prefill_token_budget == 10
    assert decision.kv_free_blocks == 96
    assert decision.kv_allocated_blocks == 4
    # ledger watermark = blocks_needed_for(prompt_len + max_new_tokens) = 4 + 2 = 6
    # (block_size=1); physically allocated so far = blocks_needed_for(prompt_len) = 4;
    # the gap (2) is the worst-case future growth already committed by the ledger.
    assert decision.expected_kv_growth_blocks == 2

    engine.step()
    second = engine.decisions[1]
    assert second.decoded == (req.request_id,)


def test_expected_kv_growth_blocks_shrinks_to_zero_as_a_sequence_decodes_to_completion():
    """FR9's 'expected KV growth' field: the ledger/physical gap should
    close to zero as a sequence's actual usage catches up to its worst-case
    watermark, then the ledger reservation itself is released on
    completion."""
    engine = SchedulerEngine(num_blocks=100, block_size=1, token_budget=10, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=1, max_new_tokens=3, target_output_len=3)])
    engine.step()  # admits: watermark=4, allocated=1 -> growth=3
    assert engine.decisions[-1].expected_kv_growth_blocks == 3
    engine.step()  # decode step 1: allocated=2 -> growth=2
    assert engine.decisions[-1].expected_kv_growth_blocks == 2
    engine.step()  # decode step 2: allocated=3 -> growth=1
    assert engine.decisions[-1].expected_kv_growth_blocks == 1
    engine.step()  # decode step 3: completes, releases the ledger -> growth=0
    assert engine.decisions[-1].expected_kv_growth_blocks == 0
    assert engine._ledger.reserved_tokens == 0


def test_decision_as_dict_is_json_serializable():
    import json

    engine = SchedulerEngine(num_blocks=10, block_size=1, token_budget=10, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=2, max_new_tokens=1, target_output_len=1)])
    engine.run_to_completion()
    for decision in engine.decisions:
        json.dumps(decision.as_dict())  # must not raise


# -- Phase 2's documented deadlock scenario is resolved here -----------------


def test_the_phase2_deadlock_scenario_completes_successfully_here():
    """Same trace/capacity as
    tests/unit/test_manager.py::test_two_sequences_can_deadlock_under_optimistic_admission
    -- Phase 2 alone deadlocks on it; the scheduler's admission ledger
    prevents the second request from ever being admitted until the first
    has completed and released, so no deadlock is possible."""
    engine = SchedulerEngine(num_blocks=3, block_size=1, token_budget=50, decode_tick=1.0)
    requests = engine.load_trace(
        [
            TraceRequestSpec(prompt_len=1, max_new_tokens=2, target_output_len=2, arrival_time=0.0),
            TraceRequestSpec(prompt_len=1, max_new_tokens=2, target_output_len=2, arrival_time=0.0),
        ]
    )
    engine.run_to_completion(max_ticks=50)  # must not raise
    assert engine.is_complete()
    assert all(r.state == RequestState.COMPLETED for r in requests)
    assert engine.event_log.of_type(EventType.DECODE_STALLED) == []
