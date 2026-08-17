"""Property tests for core invariant #7: "deterministic simulated traces
reproduce identical allocation history under fixed seed/config."

``tests/unit/test_trace.py`` only checks that ``generate_trace`` itself is
deterministic given ``(config, seed)``. That is necessary but not
sufficient: the property that actually matters for reproducible
benchmarks (FR8) is that *loading and replaying* the same materialized
trace through an engine is deterministic too -- request-id assignment,
admission ordering, and the resulting event log all have to line up,
run after run, engine after engine, in the same process. These tests
generalize that claim across many seeds/configs via hypothesis, per
spec S1.9's "deterministic decision tests" and the hypothesis dev
dependency declared specifically for this.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from minipaged.kv.contiguous import ContiguousKVEngine
from minipaged.kv.manager import PagedKVEngine
from minipaged.scheduler.scheduler import SchedulerEngine
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.trace import TraceConfig, generate_trace

_MAX_TICKS = 20_000


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    num_requests=st.integers(min_value=1, max_value=15),
    mean_interarrival=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50)
def test_replaying_the_same_trace_through_two_engines_is_byte_for_byte_identical(
    seed: int, num_requests: int, mean_interarrival: float
) -> None:
    """No KV-capacity constraint (base ``SimulationEngine``): every
    request is admitted immediately, so this isolates request-id
    assignment and event-log ordering as the things that must be
    reproducible."""
    config = TraceConfig(num_requests=num_requests, mean_interarrival=mean_interarrival)
    trace = generate_trace(config, seed=seed)

    def _replay() -> SimulationEngine:
        engine = SimulationEngine(decode_tick=1.0)
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=_MAX_TICKS)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()
    assert [seq.request_id for seq in engine_a.completed] == [
        seq.request_id for seq in engine_b.completed
    ]


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    num_requests=st.integers(min_value=1, max_value=12),
    capacity_tokens=st.integers(min_value=20, max_value=200),
)
@settings(max_examples=50)
def test_contiguous_engine_replay_is_deterministic_including_rejections(
    seed: int, num_requests: int, capacity_tokens: int
) -> None:
    """Same property, but through ``ContiguousKVEngine`` with a capacity
    tight enough to force queuing, ``ADMISSION_REJECTED`` retries, and
    (for some seeds) ``CapacityError`` cancellations -- the paths most
    likely to accidentally depend on iteration order or shared mutable
    state."""
    config = TraceConfig(num_requests=num_requests, mean_interarrival=0.5)
    trace = generate_trace(config, seed=seed)

    def _replay() -> ContiguousKVEngine:
        engine = ContiguousKVEngine(capacity_tokens=capacity_tokens, decode_tick=1.0)
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=_MAX_TICKS)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()
    assert [r.request_id for r in engine_a.cancelled] == [r.request_id for r in engine_b.cancelled]
    assert [s.request_id for s in engine_a.completed] == [s.request_id for s in engine_b.completed]


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    num_requests=st.integers(min_value=1, max_value=12),
    num_blocks=st.integers(min_value=50, max_value=100),
)
@settings(max_examples=50)
def test_paged_engine_replay_is_deterministic_including_rejections_and_stalls(
    seed: int, num_requests: int, num_blocks: int
) -> None:
    """Same property (core invariant #7) through Phase 2's ``PagedKVEngine``
    with a pool tight enough to force soft rejections and (for some
    seeds) decode-time ``DECODE_STALLED`` events -- every path that
    touches ``BlockPool``'s free-list order, which must be a
    deterministic function of call order (see ``kv/pool.py``'s
    docstring) for this property to hold at all.

    Trace ranges and ``num_blocks`` are deliberately chosen so that the
    *sum* of every request's worst-case footprint (not just its
    admission-time need) always fits total capacity -- keeping this test
    focused on determinism, not on
    ``docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md``'s
    documented (real, separately tested) deadlock limitation: with
    ``num_requests <= 12`` and each request's worst case
    ``ceil((4+4)/4) = 2`` blocks, the combined worst case (<= 24) is
    always well under ``num_blocks``'s minimum (50), so this trace can
    never actually deadlock regardless of admission timing."""
    config = TraceConfig(
        num_requests=num_requests,
        mean_interarrival=0.5,
        prompt_len_range=(1, 4),
        max_new_tokens_range=(1, 4),
    )
    trace = generate_trace(config, seed=seed)

    def _replay() -> PagedKVEngine:
        engine = PagedKVEngine(num_blocks=num_blocks, block_size=4, decode_tick=1.0)
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=_MAX_TICKS)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()
    assert [r.request_id for r in engine_a.cancelled] == [r.request_id for r in engine_b.cancelled]
    assert [s.request_id for s in engine_a.completed] == [s.request_id for s in engine_b.completed]


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    num_requests=st.integers(min_value=1, max_value=10),
    token_budget=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=50)
def test_scheduler_engine_replay_is_deterministic_including_decisions(
    seed: int, num_requests: int, token_budget: int
) -> None:
    """Same property once more through Phase 3's ``SchedulerEngine``,
    additionally asserting the scheduling timeline itself
    (``engine.decisions``) is reproducible -- not just the event log."""
    config = TraceConfig(num_requests=num_requests, mean_interarrival=0.5, prompt_len_range=(1, 8))
    trace = generate_trace(config, seed=seed)

    def _replay() -> SchedulerEngine:
        engine = SchedulerEngine(
            num_blocks=16, block_size=4, token_budget=token_budget, decode_tick=1.0
        )
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=_MAX_TICKS)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()
    assert [r.request_id for r in engine_a.cancelled] == [r.request_id for r in engine_b.cancelled]
    assert [s.request_id for s in engine_a.completed] == [s.request_id for s in engine_b.completed]
    assert [d.as_dict() for d in engine_a.decisions] == [d.as_dict() for d in engine_b.decisions]
