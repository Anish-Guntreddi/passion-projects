"""Integration tests: the full pipeline (generate_trace -> load_trace ->
run_to_completion), as opposed to the unit tests' focus on one class in
isolation. Covers spec S1.9's integration test-plan items that apply to
Phase 0-1: "trace replay" and "cancellation" (real-model smoke test and
concurrent streaming are Phase 5/6 scope and not applicable yet).
"""

from __future__ import annotations

from minipaged.kv.contiguous import ContiguousKVEngine
from minipaged.requests.request import RequestState
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceConfig, generate_trace


def test_end_to_end_trace_replay_reaches_a_terminal_state_for_every_request() -> None:
    """generate_trace -> SimulationEngine.load_trace -> run_to_completion,
    starting from nothing but a (config, seed) pair. Every request must
    end COMPLETED (the base engine never rejects, per its own contract),
    the event log must stay timestamp-monotonic, and each request's own
    event sequence must start with ARRIVED and end with exactly one
    terminal event."""
    config = TraceConfig(num_requests=30, mean_interarrival=0.7)
    trace = generate_trace(config, seed=99)

    engine = SimulationEngine(decode_tick=1.0)
    requests = engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)

    assert engine.is_complete()
    assert all(r.state == RequestState.COMPLETED for r in requests)
    assert len(engine.completed) == len(requests)
    assert engine.event_log.is_monotonic()

    for req in requests:
        types = [e.event_type for e in engine.event_log.for_request(req.request_id)]
        assert types[0] == EventType.ARRIVED
        assert types[-1] == EventType.COMPLETED
        assert types.count(EventType.ADMITTED) == 1
        assert types.count(EventType.COMPLETED) == 1


def test_end_to_end_replay_is_deterministic_across_two_full_runs() -> None:
    """Concrete, non-hypothesis regression version of core invariant #7 at
    full-pipeline scope. See tests/property/test_determinism.py for the
    hypothesis-generalized version across many seeds/configs."""
    config = TraceConfig(num_requests=25, mean_interarrival=1.2)
    trace = generate_trace(config, seed=2024)

    def _replay() -> SimulationEngine:
        engine = SimulationEngine(decode_tick=1.0)
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=10_000)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()


def test_a_request_too_large_for_total_capacity_is_cancelled_without_blocking_the_rest() -> None:
    """Cancellation, end to end: a generated trace whose every request's
    worst-case reservation exceeds total capacity must have every request
    cancelled at admission (not queued forever, not blocking anything else
    in the same trace)."""
    config = TraceConfig(
        num_requests=10,
        mean_interarrival=0.5,
        prompt_len_range=(4, 8),
        max_new_tokens_range=(4, 12),
    )
    trace = generate_trace(config, seed=13)
    # Every request's minimum possible reservation (prompt_len=4 + max_new_tokens=4
    # = 8) is already well above this capacity -- deterministically forces the
    # CapacityError/cancellation path for the whole trace.
    engine = ContiguousKVEngine(capacity_tokens=3, decode_tick=1.0)
    requests = engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)

    assert engine.is_complete()
    assert len(engine.cancelled) == len(requests)
    assert engine.completed == []
    assert all(r.state == RequestState.CANCELLED for r in requests)
    for req in requests:
        types = [e.event_type for e in engine.event_log.for_request(req.request_id)]
        assert types == [EventType.ARRIVED, EventType.ADMISSION_REJECTED, EventType.CANCELLED]
