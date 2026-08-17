"""Property test for core invariant #5 at the *scheduler* level: "scheduler
never admits work requiring more KV capacity than policy permits."

``tests/property/test_kv_memory_invariants.py`` (Phase 1) and
``tests/property/test_paged_kv_invariants.py`` (Phase 2) already prove
this for their respective allocators in isolation. This test proves the
stronger, scheduler-level claim ``docs/scheduler.md`` previewed before
Phase 2 even landed: with the admission ledger in place
(``docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md``),
Phase 2's documented deadlock scenario
(``docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md``)
is not just avoided in hand-picked test cases but *structurally
unreachable*: across many random traces, every replay completes within a
bounded number of ticks and ``DECODE_STALLED`` never appears in the event
log at all.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from minipaged.scheduler.scheduler import SchedulerEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceConfig, generate_trace

_MAX_TICKS = 2_000


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    num_requests=st.integers(min_value=1, max_value=10),
    num_blocks=st.integers(min_value=4, max_value=40),
    block_size=st.integers(min_value=1, max_value=8),
    token_budget=st.integers(min_value=2, max_value=32),
)
@settings(max_examples=100, deadline=None)
def test_scheduler_admission_never_lets_decode_exhaust_kv_capacity(
    seed: int,
    num_requests: int,
    num_blocks: int,
    block_size: int,
    token_budget: int,
) -> None:
    """Whatever trace, pool size, block size, and token budget are thrown
    at ``SchedulerEngine``: it always reaches completion within a bounded
    number of ticks (no deadlock), physical block accounting is always
    conserved (free + allocated == total, every tick), and no running
    sequence's decode step is ever stalled for lack of KV space -- the
    admission ledger's whole reason for existing."""
    config = TraceConfig(
        num_requests=num_requests,
        mean_interarrival=0.5,
        prompt_len_range=(1, max(1, block_size * 2)),
        max_new_tokens_range=(1, max(1, block_size * 2)),
    )
    trace = generate_trace(config, seed=seed)
    engine = SchedulerEngine(
        num_blocks=num_blocks, block_size=block_size, token_budget=token_budget, decode_tick=1.0
    )
    engine.load_trace(trace)
    engine.run_to_completion(max_ticks=_MAX_TICKS)  # must not raise: no deadlock

    assert engine.is_complete()
    assert engine.event_log.of_type(EventType.DECODE_STALLED) == []

    for decision in engine.decisions:
        assert decision.kv_free_blocks >= 0
        assert decision.kv_allocated_blocks >= 0
        assert decision.kv_free_blocks + decision.kv_allocated_blocks == num_blocks
        assert not decision.stalled
        assert decision.prefill_tokens_used <= decision.prefill_token_budget
        # FR9's "expected KV growth": never negative -- a sequence's physical
        # usage can never exceed its own ledger watermark (module docstring's
        # `sum(watermark_i) <= num_blocks` argument), so the ledger/physical
        # gap can never go negative either.
        assert decision.expected_kv_growth_blocks >= 0

    assert engine.manager.pool.num_free == num_blocks  # everything eventually released
