from __future__ import annotations

import pytest

from minipaged.requests.request import RequestState
from minipaged.simulation.clock import SimClock
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceRequestSpec


def _spec(
    prompt_len=4, max_new_tokens=3, target_output_len=None, arrival_time=0.0
) -> TraceRequestSpec:
    return TraceRequestSpec(
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        target_output_len=target_output_len if target_output_len is not None else max_new_tokens,
        arrival_time=arrival_time,
    )


def test_rejects_non_positive_decode_tick():
    with pytest.raises(ValueError, match="decode_tick"):
        SimulationEngine(decode_tick=0.0)


def test_load_trace_returns_requests_in_waiting_state():
    engine = SimulationEngine()
    requests = engine.load_trace([_spec(), _spec()])
    assert len(requests) == 2
    assert all(r.state == RequestState.WAITING for r in requests)
    assert not engine.is_complete()  # nothing arrived/ran yet, but the trace is loaded


def test_single_tick_admits_a_request_that_has_arrived():
    engine = SimulationEngine(decode_tick=1.0)
    [req] = engine.load_trace([_spec(arrival_time=0.0)])
    engine.step()  # clock -> 1.0, arrival_time 0.0 <= 1.0
    assert req.state == RequestState.RUNNING
    assert req.request_id in engine.running
    assert req.request_id not in engine.waiting


def test_request_stays_waiting_until_its_arrival_time():
    engine = SimulationEngine(decode_tick=1.0)
    [req] = engine.load_trace([_spec(arrival_time=5.0)])
    engine.step()  # now = 1.0 < 5.0
    assert req.state == RequestState.WAITING
    assert req.request_id in engine.waiting
    for _ in range(4):
        engine.step()
    assert req.state == RequestState.RUNNING


def test_run_to_completion_reaches_completed_for_every_request():
    engine = SimulationEngine(decode_tick=1.0)
    requests = engine.load_trace(
        [_spec(max_new_tokens=2), _spec(max_new_tokens=5, arrival_time=1.0)]
    )
    engine.run_to_completion()
    assert engine.is_complete()
    assert all(r.state == RequestState.COMPLETED for r in requests)
    assert len(engine.completed) == 2


def test_run_to_completion_raises_past_max_ticks():
    engine = SimulationEngine(decode_tick=1.0)
    engine.load_trace([_spec(arrival_time=1000.0)])
    with pytest.raises(RuntimeError, match="max_ticks"):
        engine.run_to_completion(max_ticks=5)


def test_admitted_and_prefill_done_events_recorded_at_admission():
    engine = SimulationEngine(decode_tick=1.0)
    [req] = engine.load_trace([_spec(arrival_time=0.0)])
    engine.step()
    events = engine.event_log.for_request(req.request_id)
    types = [e.event_type for e in events]
    assert types == [EventType.ARRIVED, EventType.ADMITTED, EventType.PREFILL_DONE]


def test_decode_step_events_carry_running_token_count():
    engine = SimulationEngine(decode_tick=1.0)
    [req] = engine.load_trace([_spec(max_new_tokens=3, target_output_len=3, arrival_time=0.0)])
    engine.run_to_completion()
    decode_events = [
        e
        for e in engine.event_log.for_request(req.request_id)
        if e.event_type == EventType.DECODE_STEP
    ]
    assert [e.detail["tokens_generated"] for e in decode_events] == [1, 2, 3]


def test_base_engine_never_rejects_or_cancels():
    engine = SimulationEngine(decode_tick=1.0)
    engine.load_trace([_spec() for _ in range(5)])
    engine.run_to_completion()
    assert engine.cancelled == []
    assert engine.event_log.of_type(EventType.ADMISSION_REJECTED) == []


def test_uses_the_provided_clock_instance():
    clock = SimClock(start=100.0)
    engine = SimulationEngine(clock=clock, decode_tick=2.0)
    engine.load_trace([_spec(arrival_time=100.0)])
    engine.step()
    assert engine.clock is clock
    assert clock.now() == 102.0


def test_completed_sequences_are_removed_from_running():
    engine = SimulationEngine(decode_tick=1.0)
    engine.load_trace([_spec(max_new_tokens=1, target_output_len=1, arrival_time=0.0)])
    engine.run_to_completion()
    assert engine.running == {}
    assert len(engine.completed) == 1
