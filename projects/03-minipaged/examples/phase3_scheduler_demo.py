#!/usr/bin/env python3
"""Phase 3 exit-criterion demo: replay a synthetic arrival trace through
``SchedulerEngine`` and print the resulting scheduling timeline --
``SchedulerDecision`` per tick (FR9) -- alongside the same "does paged
admission alone deadlock here" comparison Decision 0004/0005 describe.

    python examples/phase3_scheduler_demo.py
"""

from __future__ import annotations

from minipaged.scheduler.scheduler import SchedulerEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceConfig, generate_trace


def main() -> None:
    config = TraceConfig(
        num_requests=20,
        mean_interarrival=0.4,
        prompt_len_range=(4, 24),
        max_new_tokens_range=(8, 48),
        completion_fraction_range=(0.4, 1.0),
    )
    seed = 11
    # token_budget (32) is kept comfortably above prompt_len_range's max
    # (24) so every request's prompt fits within a single tick's budget in
    # principle -- cancellations below are only ever the KV-capacity kind,
    # not budget-too-small-to-ever-admit-this-prompt.
    num_blocks, block_size, token_budget = 60, 4, 32
    trace = generate_trace(config, seed=seed)

    print(
        f"Phase 3 scheduler demo -- {config.num_requests} requests, seed={seed}, "
        f"{num_blocks} blocks x {block_size} tokens, token_budget={token_budget}\n"
    )

    engine = SchedulerEngine(
        num_blocks=num_blocks, block_size=block_size, token_budget=token_budget, decode_tick=1.0
    )
    engine.load_trace(trace)
    engine.run_to_completion(max_ticks=10_000)

    header = (
        f"{'t':>5}  {'admit':>5}  {'decode':>6}  {'reject':>6}  {'cancel':>6}  "
        f"{'done':>4}  {'prefill':>7}  {'free':>4}  {'alloc':>5}  {'growth':>6}"
    )
    print(header)
    print("-" * len(header))
    for d in engine.decisions:
        print(
            f"{d.tick:5.0f}  {len(d.admitted):5d}  {len(d.decoded):6d}  {len(d.rejected):6d}  "
            f"{len(d.cancelled):6d}  {len(d.completed):4d}  "
            f"{d.prefill_tokens_used:3d}/{d.prefill_token_budget:<3d}  "
            f"{d.kv_free_blocks:4d}  {d.kv_allocated_blocks:5d}  {d.expected_kv_growth_blocks:6d}"
        )

    stalled = engine.event_log.of_type(EventType.DECODE_STALLED)
    print(
        f"\nCompleted: {len(engine.completed)}   Cancelled: {len(engine.cancelled)}   "
        f"Ticks: {len(engine.decisions)}   DECODE_STALLED events: {len(stalled)} "
        f"(always 0 -- the admission ledger's guarantee, see docs/decisions/0005-...md)"
    )
    peak_running = max(d.kv_allocated_blocks for d in engine.decisions)
    print(f"Peak allocated blocks: {peak_running}/{num_blocks}")
    assert stalled == [], "the scheduler's admission ledger should make this impossible"


if __name__ == "__main__":
    main()
