from __future__ import annotations

import pytest

from minipaged.kv.contiguous import (
    CapacityError,
    ContiguousKVEngine,
    ContiguousMemoryMetrics,
    ContiguousMemoryModel,
)
from minipaged.requests.request import RequestState
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceConfig, TraceRequestSpec, generate_trace


def _spec(
    prompt_len=4, max_new_tokens=3, target_output_len=None, arrival_time=0.0
) -> TraceRequestSpec:
    return TraceRequestSpec(
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        target_output_len=target_output_len if target_output_len is not None else max_new_tokens,
        arrival_time=arrival_time,
    )


# -- ContiguousMemoryModel: reservation bookkeeping --------------------------


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity_tokens"):
        ContiguousMemoryModel(capacity_tokens=0)


def test_try_reserve_commits_when_capacity_is_available():
    model = ContiguousMemoryModel(capacity_tokens=10)
    assert model.try_reserve("req-1", size=6, used=4) is True
    assert model.is_reserved("req-1")
    assert model.reserved_tokens == 6
    assert model.used_tokens == 4
    assert model.free_tokens == 4


def test_try_reserve_returns_false_when_free_capacity_insufficient():
    model = ContiguousMemoryModel(capacity_tokens=10)
    assert model.try_reserve("req-1", size=8, used=8) is True
    assert model.try_reserve("req-2", size=5, used=5) is False
    assert not model.is_reserved("req-2")
    assert model.reserved_tokens == 8  # unchanged by the failed attempt


def test_try_reserve_raises_capacity_error_when_size_exceeds_total_capacity():
    model = ContiguousMemoryModel(capacity_tokens=10)
    with pytest.raises(CapacityError, match="req-1"):
        model.try_reserve("req-1", size=11, used=0)
    assert not model.is_reserved("req-1")


def test_try_reserve_rejects_used_exceeding_size():
    model = ContiguousMemoryModel(capacity_tokens=10)
    with pytest.raises(ValueError, match="exceeds requested size"):
        model.try_reserve("req-1", size=5, used=6)


def test_try_reserve_rejects_duplicate_reservation():
    model = ContiguousMemoryModel(capacity_tokens=10)
    model.try_reserve("req-1", size=4, used=4)
    with pytest.raises(ValueError, match="already holds a reservation"):
        model.try_reserve("req-1", size=4, used=4)


def test_update_used_tracks_growth():
    model = ContiguousMemoryModel(capacity_tokens=10)
    model.try_reserve("req-1", size=8, used=4)
    model.update_used("req-1", 6)
    assert model.used_tokens == 6


def test_update_used_rejects_exceeding_reservation():
    model = ContiguousMemoryModel(capacity_tokens=10)
    model.try_reserve("req-1", size=8, used=4)
    with pytest.raises(ValueError, match="exceeds its own reservation"):
        model.update_used("req-1", 9)


def test_release_frees_the_reservation_slots():
    model = ContiguousMemoryModel(capacity_tokens=10)
    model.try_reserve("req-1", size=6, used=6)
    freed = model.release("req-1")
    assert freed == 6
    assert not model.is_reserved("req-1")
    assert model.reserved_tokens == 0
    assert model.free_tokens == 10


def test_snapshot_reports_waste_and_utilization():
    model = ContiguousMemoryModel(capacity_tokens=20)
    model.try_reserve("req-1", size=10, used=4)  # 6 wasted
    model.try_reserve("req-2", size=5, used=5)  # 0 wasted
    metrics = model.snapshot()
    assert isinstance(metrics, ContiguousMemoryMetrics)
    assert metrics.capacity == 20
    assert metrics.reserved == 15
    assert metrics.used == 9
    assert metrics.free == 5
    assert metrics.waste == 6
    assert metrics.utilization == pytest.approx(9 / 20)
    assert metrics.waste_ratio == pytest.approx(6 / 20)
    assert metrics.as_dict()["waste"] == 6


# -- ContiguousKVEngine: admission gated on capacity -------------------------


def test_admission_reserves_worst_case_prompt_plus_max_new_tokens():
    engine = ContiguousKVEngine(capacity_tokens=100, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=4, max_new_tokens=3, arrival_time=0.0)])
    engine.step()
    assert engine.memory.reserved_tokens == 7  # prompt_len + max_new_tokens
    assert engine.memory.used_tokens == 4  # prompt_len, before any decode step


def test_admission_is_rejected_when_capacity_is_insufficient():
    # req-1 reserves 8 of 10 slots; req-2 needs 5 more, which do not fit
    # while req-1 is still running, so it stays queued (not cancelled).
    engine = ContiguousKVEngine(capacity_tokens=10, decode_tick=1.0)
    [req1, req2] = engine.load_trace(
        [
            _spec(prompt_len=4, max_new_tokens=4, arrival_time=0.0),
            _spec(prompt_len=3, max_new_tokens=2, arrival_time=0.0),
        ]
    )
    engine.step()
    assert req1.state == RequestState.RUNNING
    assert req2.state == RequestState.WAITING
    assert req2.request_id in engine.waiting

    rejected = engine.event_log.for_request(req2.request_id)
    assert [e.event_type for e in rejected] == [
        EventType.ARRIVED,
        EventType.ADMISSION_REJECTED,
    ]

    # Once req-1 completes and releases its reservation, req-2 fits.
    engine.run_to_completion()
    assert req2.state == RequestState.COMPLETED
    assert req2.request_id not in engine.cancelled


def test_request_larger_than_total_capacity_raises_capacity_error_at_admission():
    engine = ContiguousKVEngine(capacity_tokens=5, decode_tick=1.0)
    [req] = engine.load_trace([_spec(prompt_len=4, max_new_tokens=4, arrival_time=0.0)])
    engine.step()
    assert req.state == RequestState.CANCELLED
    assert req in engine.cancelled
    assert not engine.memory.is_reserved(req.request_id)

    types = [e.event_type for e in engine.event_log.for_request(req.request_id)]
    assert types == [EventType.ARRIVED, EventType.ADMISSION_REJECTED, EventType.CANCELLED]


def test_used_tracks_decode_progress_and_release_frees_on_completion():
    engine = ContiguousKVEngine(capacity_tokens=20, decode_tick=1.0)
    engine.load_trace(
        [_spec(prompt_len=4, max_new_tokens=3, target_output_len=2, arrival_time=0.0)]
    )
    engine.run_to_completion()
    assert engine.memory.reserved_tokens == 0  # released on completion
    assert engine.memory.used_tokens == 0
    assert len(engine.completed) == 1


# -- Phase 1 exit criterion: variable-length trace demonstrates waste --------


def test_variable_length_trace_demonstrates_nonzero_fragmentation_waste():
    """The actual Phase 1 exit criterion, made an explicit regression test:
    a variable-length trace with early completion produces measurable,
    nonzero waste at some point during replay, and zero waste once every
    sequence has completed and released its reservation."""
    config = TraceConfig(
        num_requests=12,
        mean_interarrival=0.5,
        prompt_len_range=(4, 16),
        max_new_tokens_range=(8, 64),
        completion_fraction_range=(0.3, 0.7),  # guarantees early completion
    )
    trace = generate_trace(config, seed=42)
    engine = ContiguousKVEngine(capacity_tokens=100_000, decode_tick=1.0)
    engine.load_trace(trace)

    observed_waste: list[int] = []
    while not engine.is_complete():
        engine.step()
        observed_waste.append(engine.memory.snapshot().waste)

    assert max(observed_waste) > 0
    assert all(w >= 0 for w in observed_waste)  # waste never goes negative
    # Every sequence completed and released: nothing left reserved.
    assert engine.memory.reserved_tokens == 0
    assert engine.memory.snapshot().waste == 0
    assert len(engine.completed) == len(trace)
    assert engine.cancelled == []


def test_waste_reaches_zero_when_a_sequence_uses_its_full_reservation():
    """Contrast case for the model-level waste computation: a sequence
    that eventually generates exactly its committed max_new_tokens budget
    (no early completion) has reserved == used right before release --
    zero waste, since the worst-case reservation was fully used. Compare
    with the early-completion case just below, where it is not."""
    model = ContiguousMemoryModel(capacity_tokens=20)
    model.try_reserve("req-1", size=9, used=4)  # prompt_len=4, max_new_tokens=5
    for used in (5, 6, 7, 8, 9):  # generates all 5 tokens of its budget
        model.update_used("req-1", used)
    assert model.snapshot().waste == 0


def test_waste_stays_positive_when_a_sequence_completes_early():
    model = ContiguousMemoryModel(capacity_tokens=20)
    model.try_reserve("req-1", size=9, used=4)  # prompt_len=4, max_new_tokens=5
    model.update_used("req-1", 6)  # "EOS" after only 2 of its 5-token budget
    assert model.snapshot().waste == 3  # reserved 9, used 6
