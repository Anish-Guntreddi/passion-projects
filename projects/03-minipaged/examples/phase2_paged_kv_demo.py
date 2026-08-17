#!/usr/bin/env python3
"""Phase 2 exit-criterion demo: the *same* trace and total token capacity
Phase 1's fragmentation demo used, replayed through the paged allocator
instead of the contiguous baseline, printing utilization statistics --
and, side by side, exactly what paging saves versus Phase 1's contiguous
reservation on identical inputs.

    python examples/phase2_paged_kv_demo.py
"""

from __future__ import annotations

from minipaged.kv.contiguous import ContiguousKVEngine
from minipaged.kv.manager import PagedKVEngine
from minipaged.simulation.trace import TraceConfig, generate_trace


def main() -> None:
    config = TraceConfig(
        num_requests=16,
        mean_interarrival=0.6,
        prompt_len_range=(4, 32),
        max_new_tokens_range=(8, 96),
        completion_fraction_range=(0.3, 0.8),
    )
    seed = 7
    capacity_tokens = 400
    block_size = 16
    num_blocks = capacity_tokens // block_size
    trace = generate_trace(config, seed=seed)

    print(
        f"Phase 2 paged-vs-contiguous demo -- {config.num_requests} requests, seed={seed}, "
        f"capacity={capacity_tokens} token-slots ({num_blocks} blocks x {block_size} tokens)\n"
    )

    contiguous = ContiguousKVEngine(capacity_tokens=capacity_tokens, decode_tick=1.0)
    contiguous.load_trace(trace)
    contiguous_peak_waste_ratio = 0.0
    while not contiguous.is_complete():
        contiguous.step()
        contiguous_peak_waste_ratio = max(
            contiguous_peak_waste_ratio, contiguous.memory.snapshot().waste_ratio
        )

    paged = PagedKVEngine(num_blocks=num_blocks, block_size=block_size, decode_tick=1.0)
    paged.load_trace(trace)

    header = f"{'t':>6}  {'blocks':>6}  {'free':>6}  {'used':>6}  {'waste':>6}  {'util%':>7}  {'waste%':>7}"
    print(header)
    print("-" * len(header))

    paged_peak_waste_ratio = 0.0
    while not paged.is_complete():
        paged.step()
        snap = paged.manager.snapshot()
        paged_peak_waste_ratio = max(paged_peak_waste_ratio, snap.waste_ratio)
        print(
            f"{paged.clock.now():6.1f}  {snap.allocated_blocks:6d}  {snap.free_blocks:6d}  "
            f"{snap.used_tokens:6d}  {snap.waste:6d}  {snap.utilization * 100:6.1f}%  "
            f"{snap.waste_ratio * 100:6.1f}%"
        )

    print(
        f"\nPeak waste ratio -- contiguous: {contiguous_peak_waste_ratio * 100:5.1f}%   "
        f"paged: {paged_peak_waste_ratio * 100:5.1f}%   "
        f"(same trace, same {capacity_tokens}-token-slot total capacity)"
    )
    print(
        "Paging never commits to a sequence's max_new_tokens budget up front -- only to whole "
        "blocks covering what it has actually generated -- which is exactly the difference above."
    )
    print(
        f"Completed: {len(paged.completed)}   Cancelled: {len(paged.cancelled)}   "
        f"(contiguous: {len(contiguous.completed)} completed, {len(contiguous.cancelled)} cancelled)"
    )
    assert paged_peak_waste_ratio < contiguous_peak_waste_ratio, (
        "expected paging to waste strictly less than the contiguous baseline on this trace"
    )


if __name__ == "__main__":
    main()
