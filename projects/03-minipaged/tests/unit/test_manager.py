from __future__ import annotations

import pytest

from minipaged.kv.manager import PagedKVEngine, PagedKVManager, PagedKVMetrics
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


# -- PagedKVManager: block-table bookkeeping ---------------------------------


def test_admit_allocates_blocks_rounded_up_for_prompt_len_only():
    manager = PagedKVManager(num_blocks=10, block_size=4)
    assert manager.admit("req-1", prompt_len=10) is True  # needs 3 blocks (ceil(10/4))
    assert manager.is_admitted("req-1")
    assert manager.pool.num_allocated == 3
    assert manager.pool.num_free == 7


def test_admit_does_not_reserve_max_new_tokens():
    """The whole point of paging: admission needs only prompt_len blocks,
    not prompt_len + max_new_tokens (contrast with Phase 1's
    ContiguousMemoryModel, which must reserve the worst case)."""
    manager = PagedKVManager(num_blocks=2, block_size=16)
    assert manager.admit("req-1", prompt_len=10) is True  # 1 block; max_new_tokens irrelevant here
    assert manager.pool.num_allocated == 1


def test_admit_returns_false_when_free_blocks_insufficient():
    manager = PagedKVManager(num_blocks=2, block_size=4)
    assert manager.admit("req-1", prompt_len=8) is True  # 2 blocks, pool now full
    assert manager.admit("req-2", prompt_len=1) is False  # 1 block needed, 0 free
    assert not manager.is_admitted("req-2")
    assert manager.pool.num_allocated == 2  # unchanged by the failed attempt


def test_admit_rejects_duplicate_request_id():
    manager = PagedKVManager(num_blocks=10, block_size=4)
    manager.admit("req-1", prompt_len=4)
    with pytest.raises(ValueError, match="already admitted"):
        manager.admit("req-1", prompt_len=4)


def test_can_grow_to_is_a_pure_check():
    manager = PagedKVManager(num_blocks=2, block_size=4)
    manager.admit("req-1", prompt_len=4)  # 1 block, 1 free left
    assert manager.can_grow_to("req-1", num_tokens=8) is True  # needs 1 more, 1 free
    assert manager.can_grow_to("req-1", num_tokens=12) is False  # needs 2 more, only 1 free
    assert manager.pool.num_free == 1  # unchanged -- pure check


def test_grow_extends_the_block_table_lazily():
    manager = PagedKVManager(num_blocks=10, block_size=4)
    manager.admit("req-1", prompt_len=4)  # 1 block
    assert manager.grow("req-1", num_tokens=4) == 0  # still fits
    assert manager.grow("req-1", num_tokens=5) == 1  # crosses the boundary
    assert manager.pool.num_allocated == 2


def test_release_frees_every_block_the_request_held():
    manager = PagedKVManager(num_blocks=10, block_size=4)
    manager.admit("req-1", prompt_len=10)  # 3 blocks
    freed = manager.release("req-1")
    assert freed == 3
    assert not manager.is_admitted("req-1")
    assert manager.pool.num_free == 10


def test_snapshot_reports_block_quantized_waste_and_utilization():
    manager = PagedKVManager(num_blocks=10, block_size=4)
    manager.admit("req-1", prompt_len=5)  # 2 blocks (8 tokens reserved), 5 used -> waste 3
    manager.admit("req-2", prompt_len=4)  # 1 block (4 tokens reserved), 4 used -> waste 0
    metrics = manager.snapshot()
    assert isinstance(metrics, PagedKVMetrics)
    assert metrics.num_blocks == 10
    assert metrics.block_size == 4
    assert metrics.allocated_blocks == 3
    assert metrics.free_blocks == 7
    assert metrics.capacity_tokens == 40
    assert metrics.reserved_tokens == 12
    assert metrics.used_tokens == 9
    assert metrics.waste == 3
    assert metrics.utilization == pytest.approx(9 / 40)
    assert metrics.waste_ratio == pytest.approx(3 / 40)
    assert metrics.as_dict()["waste"] == 3


# -- PagedKVEngine: admission + decode-time growth gated on paged capacity --


def test_admission_allocates_only_prompt_blocks_not_worst_case():
    engine = PagedKVEngine(num_blocks=100, block_size=4, decode_tick=1.0)
    engine.load_trace([_spec(prompt_len=4, max_new_tokens=100, arrival_time=0.0)])
    engine.step()
    assert engine.manager.pool.num_allocated == 1  # ceil(4/4), not ceil(104/4)


def test_admission_is_soft_rejected_when_free_blocks_insufficient():
    # num_blocks=3: large enough that both requests' own worst case
    # (prompt_len + max_new_tokens, rounded up) fits total capacity, so
    # neither is hard-cancelled -- req1's admission-time need (2 blocks)
    # just happens to leave only 1 free, less than req2's admission-time
    # need (2 blocks), forcing a genuine soft (retryable) rejection.
    engine = PagedKVEngine(num_blocks=3, block_size=4, decode_tick=1.0)
    [req1, req2] = engine.load_trace(
        [
            _spec(prompt_len=8, max_new_tokens=1, arrival_time=0.0),  # needs 2 blocks; worst 3<=3
            _spec(prompt_len=5, max_new_tokens=1, arrival_time=0.0),  # needs 2 blocks; worst 2<=3
        ]
    )
    engine.step()
    assert req1.state == RequestState.RUNNING
    assert req2.state == RequestState.WAITING
    rejected = engine.event_log.for_request(req2.request_id)
    assert [e.event_type for e in rejected] == [EventType.ARRIVED, EventType.ADMISSION_REJECTED]

    engine.run_to_completion()
    assert req2.state == RequestState.COMPLETED
    assert req2.request_id not in engine.cancelled


def test_request_whose_worst_case_exceeds_total_capacity_is_cancelled():
    engine = PagedKVEngine(num_blocks=2, block_size=4, decode_tick=1.0)
    [req] = engine.load_trace(
        [_spec(prompt_len=4, max_new_tokens=20, arrival_time=0.0)]  # worst case: 6 blocks > 2 total
    )
    engine.step()
    assert req.state == RequestState.CANCELLED
    assert req in engine.cancelled
    assert not engine.manager.is_admitted(req.request_id)
    types = [e.event_type for e in engine.event_log.for_request(req.request_id)]
    assert types == [EventType.ARRIVED, EventType.ADMISSION_REJECTED, EventType.CANCELLED]


def test_decode_grows_the_block_table_and_release_frees_on_completion():
    engine = PagedKVEngine(num_blocks=10, block_size=4, decode_tick=1.0)
    engine.load_trace(
        [_spec(prompt_len=4, max_new_tokens=3, target_output_len=2, arrival_time=0.0)]
    )
    engine.run_to_completion()
    assert engine.manager.pool.num_free == 10  # released on completion
    assert len(engine.completed) == 1


def test_variable_length_trace_produces_utilization_statistics():
    """Phase 2 exit criterion, as a regression test: the same kind of
    variable-length trace Phase 1 uses replays cleanly through the paged
    engine and produces a non-trivial utilization snapshot at every tick."""
    config = TraceConfig(
        num_requests=12,
        mean_interarrival=0.5,
        prompt_len_range=(4, 16),
        max_new_tokens_range=(8, 64),
        completion_fraction_range=(0.3, 0.7),
    )
    trace = generate_trace(config, seed=42)
    engine = PagedKVEngine(num_blocks=1000, block_size=16, decode_tick=1.0)
    engine.load_trace(trace)

    snapshots = []
    while not engine.is_complete():
        engine.step()
        snapshots.append(engine.manager.snapshot())

    assert snapshots  # at least one tick happened
    assert all(0.0 <= s.utilization <= 1.0 for s in snapshots)
    assert all(s.waste >= 0 for s in snapshots)
    assert engine.manager.pool.num_free == 1000  # everything released by the end
    assert len(engine.completed) == len(trace)
    assert engine.cancelled == []


def test_paged_allocator_wastes_less_than_contiguous_on_the_same_trace():
    """The actual point of Phase 2, made an explicit comparison: on the
    identical trace and identical total token capacity, the paged
    allocator's peak waste ratio is lower than the contiguous baseline's,
    because it never commits to a sequence's max_new_tokens budget up
    front -- only to whole blocks covering what it has actually
    generated."""
    from minipaged.kv.contiguous import ContiguousKVEngine

    config = TraceConfig(
        num_requests=16,
        mean_interarrival=0.5,
        prompt_len_range=(4, 32),
        max_new_tokens_range=(8, 96),
        completion_fraction_range=(0.3, 0.8),
    )
    trace = generate_trace(config, seed=7)
    capacity_tokens = 400
    block_size = 16

    contiguous = ContiguousKVEngine(capacity_tokens=capacity_tokens, decode_tick=1.0)
    contiguous.load_trace(trace)
    contiguous_peak_waste_ratio = 0.0
    while not contiguous.is_complete():
        contiguous.step()
        contiguous_peak_waste_ratio = max(
            contiguous_peak_waste_ratio, contiguous.memory.snapshot().waste_ratio
        )

    paged = PagedKVEngine(
        num_blocks=capacity_tokens // block_size, block_size=block_size, decode_tick=1.0
    )
    paged.load_trace(trace)
    paged_peak_waste_ratio = 0.0
    while not paged.is_complete():
        paged.step()
        paged_peak_waste_ratio = max(paged_peak_waste_ratio, paged.manager.snapshot().waste_ratio)

    assert paged_peak_waste_ratio < contiguous_peak_waste_ratio
    assert len(paged.completed) == len(trace)
    assert paged.cancelled == []


# -- Documented Phase 2 limitation: optimistic admission can deadlock -------


def test_two_sequences_can_deadlock_under_optimistic_admission():
    """Documented in docs/decisions/0004: admitting on current need alone
    (not worst case) means two sequences can each be admitted, and each
    individually has a worst-case footprint that fits total capacity (so
    neither is hard-cancelled at admission) -- but together their combined
    eventual need exceeds the pool. Both grow far enough to consume every
    remaining free block, then both simultaneously need one more block
    with none free: neither can ever progress or complete, so neither
    ever releases, forever. run_to_completion's max_ticks safety net turns
    that into a fast, deterministic test failure (RuntimeError) rather
    than a real hang. Phase 3's SchedulerEngine fixes this exact scenario
    -- see tests/unit/test_scheduler.py."""
    # 3 blocks total, block_size=1. Each request's own worst case
    # (prompt_len + max_new_tokens = 3) exactly fits alone -- not
    # hard-cancelled -- but admitting *both* (2 blocks used up front)
    # leaves only 1 free for two sequences that will jointly want 2 more.
    engine = PagedKVEngine(num_blocks=3, block_size=1, decode_tick=1.0)
    engine.load_trace(
        [
            TraceRequestSpec(prompt_len=1, max_new_tokens=2, target_output_len=2, arrival_time=0.0),
            TraceRequestSpec(prompt_len=1, max_new_tokens=2, target_output_len=2, arrival_time=0.0),
        ]
    )
    with pytest.raises(RuntimeError, match="max_ticks"):
        engine.run_to_completion(max_ticks=50)

    # Both sequences are stuck RUNNING, unable to grow far enough to reach
    # their target -- not crashed, not corrupted, just permanently stalled.
    assert len(engine.running) == 2
    assert engine.manager.pool.num_free == 0
    stalled_events = engine.event_log.of_type(EventType.DECODE_STALLED)
    assert len(stalled_events) >= 2  # both sequences stalled at least once
