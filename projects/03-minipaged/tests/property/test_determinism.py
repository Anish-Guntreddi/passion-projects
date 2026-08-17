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
