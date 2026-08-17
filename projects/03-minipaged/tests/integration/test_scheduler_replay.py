"""Integration test for Phase 3's exit criterion: "Runtime replays
arrivals and produces scheduling timelines." End to end:
generate_trace -> load_trace -> run_to_completion -> inspect
``engine.decisions`` (the scheduling timeline, FR9's ``SchedulerDecision``
accumulated per tick) -- as opposed to the unit tests' focus on individual
scheduling decisions in isolation.
"""

from __future__ import annotations

from minipaged.requests.request import RequestState
from minipaged.scheduler.scheduler import SchedulerEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceConfig, generate_trace


def test_end_to_end_replay_produces_a_scheduling_timeline_covering_every_request() -> None:
    config = TraceConfig(
        num_requests=20,
        mean_interarrival=0.7,
        prompt_len_range=(2, 16),
        max_new_tokens_range=(2, 24),
    )
    trace = generate_trace(config, seed=101)

    engine = SchedulerEngine(num_blocks=64, block_size=4, token_budget=24, decode_tick=1.0)
    requests = engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)

    assert engine.is_complete()
    assert all(r.state in (RequestState.COMPLETED, RequestState.CANCELLED) for r in requests)
    assert engine.event_log.is_monotonic()

    # The scheduling timeline: one SchedulerDecision per tick that actually
    # ran, ticks strictly increasing, and every request that was ever
    # admitted appears in exactly one tick's `admitted` tuple.
    assert engine.decisions
    ticks = [d.tick for d in engine.decisions]
    assert ticks == sorted(ticks)
    assert len(ticks) == len(set(ticks))  # one decision per distinct tick

    admitted_ids = [rid for d in engine.decisions for rid in d.admitted]
    assert len(admitted_ids) == len(set(admitted_ids))  # each request admitted at most once
    completed_ids = {r.request_id for r in requests if r.state == RequestState.COMPLETED}
    cancelled_ids = {r.request_id for r in requests if r.state == RequestState.CANCELLED}
    assert completed_ids.issubset(set(admitted_ids))  # every completed request was admitted first
    assert cancelled_ids.isdisjoint(set(admitted_ids))  # a cancelled request is never admitted

    timeline_completed_ids = [rid for d in engine.decisions for rid in d.completed]
    assert set(timeline_completed_ids) == completed_ids


def test_end_to_end_replay_is_deterministic_across_two_full_runs() -> None:
    config = TraceConfig(num_requests=15, mean_interarrival=1.0, prompt_len_range=(1, 12))
    trace = generate_trace(config, seed=2025)

    def _replay() -> SchedulerEngine:
        engine = SchedulerEngine(num_blocks=48, block_size=4, token_budget=16, decode_tick=1.0)
        engine.load_trace(trace)
        engine.run_to_completion(max_ticks=10_000)
        return engine

    engine_a = _replay()
    engine_b = _replay()

    assert engine_a.event_log.as_dicts() == engine_b.event_log.as_dicts()
    assert [d.as_dict() for d in engine_a.decisions] == [d.as_dict() for d in engine_b.decisions]


def test_continuous_batching_is_observable_in_the_timeline() -> None:
    """The batch is built up and drained gradually -- not every request
    admitted in a single tick, and admission continues to happen on
    later ticks after some earlier requests have already completed."""
    config = TraceConfig(num_requests=10, mean_interarrival=0.2, prompt_len_range=(4, 12))
    trace = generate_trace(config, seed=55)
    engine = SchedulerEngine(num_blocks=20, block_size=4, token_budget=8, decode_tick=1.0)
    engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)

    ticks_with_admissions = [d.tick for d in engine.decisions if d.admitted]
    ticks_with_completions = [d.tick for d in engine.decisions if d.completed]
    assert len(ticks_with_admissions) > 1  # not all admitted in one shot
    assert ticks_with_completions  # at least some requests finished
    # At least one admission happens strictly after the earliest completion --
    # i.e. the batch keeps admitting new work as old work finishes, the
    # actual definition of continuous batching (FR4).
    assert max(ticks_with_admissions) > min(ticks_with_completions)


def test_kv_exhaustion_never_reaches_the_engine_layer() -> None:
    """S1.9 failure-mode item ("KV exhaustion") is exercised here under a
    deliberately tight pool relative to demand -- forcing plenty of soft
    rejections and deferred admission -- and the scheduler's admission
    ledger guarantees no DECODE_STALLED ever occurs, in contrast with
    Phase 2's PagedKVEngine on the same kind of trace (see
    tests/unit/test_manager.py's deadlock test)."""
    config = TraceConfig(
        num_requests=25,
        mean_interarrival=0.1,  # bursty arrivals -- heavy admission pressure
        prompt_len_range=(4, 16),
        max_new_tokens_range=(4, 32),
    )
    trace = generate_trace(config, seed=909)
    engine = SchedulerEngine(num_blocks=16, block_size=4, token_budget=8, decode_tick=1.0)
    engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)  # must not raise (no deadlock)

    assert engine.is_complete()
    assert engine.event_log.of_type(EventType.DECODE_STALLED) == []
    assert any(d.rejected for d in engine.decisions)  # the pressure was real, not vacuous
