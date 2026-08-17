#!/usr/bin/env python3
"""Phase 1 exit-criterion demo: a variable-length trace demonstrates
fragmentation/waste numerically for the contiguous KV baseline.

Generates a deterministic variable-length trace with early completion
(target_output_len < max_new_tokens for most requests -- see
docs/kv-memory.md for why that specific combination is what produces
measurable waste), replays it through ``ContiguousKVEngine``, and prints
the memory snapshot (reserved/used/free/waste/utilization) at every tick.
Run with:

    python examples/phase1_fragmentation_demo.py
"""

from __future__ import annotations

from minipaged.kv.contiguous import ContiguousKVEngine
from minipaged.simulation.trace import TraceConfig, generate_trace


def main() -> None:
    config = TraceConfig(
        num_requests=16,
        mean_interarrival=0.6,
        prompt_len_range=(4, 32),
        max_new_tokens_range=(8, 96),
        completion_fraction_range=(0.3, 0.8),  # most sequences stop before their budget
    )
    seed = 7
    capacity_tokens = 400
    trace = generate_trace(config, seed=seed)

    print(
        f"Phase 1 fragmentation demo -- {config.num_requests} requests, "
        f"seed={seed}, capacity={capacity_tokens} token-slots\n"
    )

    engine = ContiguousKVEngine(capacity_tokens=capacity_tokens, decode_tick=1.0)
    engine.load_trace(trace)

    header = f"{'t':>6}  {'reserved':>8}  {'used':>6}  {'free':>6}  {'waste':>6}  {'util%':>7}  {'waste%':>7}"
    print(header)
    print("-" * len(header))

    peak_waste = 0
    peak_waste_ratio = 0.0
    while not engine.is_complete():
        engine.step()
        snap = engine.memory.snapshot()
        peak_waste = max(peak_waste, snap.waste)
        peak_waste_ratio = max(peak_waste_ratio, snap.waste_ratio)
        print(
            f"{engine.clock.now():6.1f}  {snap.reserved:8d}  {snap.used:6d}  "
            f"{snap.free:6d}  {snap.waste:6d}  {snap.utilization * 100:6.1f}%  "
            f"{snap.waste_ratio * 100:6.1f}%"
        )

    print(
        f"\nPeak waste: {peak_waste} token-slots ({peak_waste_ratio * 100:.1f}% of capacity) "
        f"-- this is memory the contiguous allocator held but nothing was using, "
        f"and nobody else could borrow. Phase 2's paged allocator exists to eliminate it."
    )
    print(f"Completed: {len(engine.completed)}   Cancelled: {len(engine.cancelled)}")
    assert peak_waste > 0, "expected the demo trace to produce measurable waste"


if __name__ == "__main__":
    main()
