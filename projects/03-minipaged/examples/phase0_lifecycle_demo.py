#!/usr/bin/env python3
"""Phase 0 exit-criterion demo: synthetic requests move
waiting -> running -> completed without a model.

Generates a small deterministic trace, replays it through the base
``SimulationEngine`` (no KV-capacity constraint), and prints every
recorded event plus a summary -- the runnable evidence that the Phase 0
lifecycle loop works end to end. Run with:

    python examples/phase0_lifecycle_demo.py
"""

from __future__ import annotations

from minipaged.requests.request import RequestState
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.trace import TraceConfig, generate_trace


def main() -> None:
    config = TraceConfig(
        num_requests=8,
        mean_interarrival=1.5,
        prompt_len_range=(4, 16),
        max_new_tokens_range=(3, 10),
    )
    seed = 1
    trace = generate_trace(config, seed=seed)

    print(f"Phase 0 lifecycle demo -- {config.num_requests} requests, seed={seed}\n")

    engine = SimulationEngine(decode_tick=1.0)
    requests = engine.load_trace(trace)
    engine.run_to_completion()

    print("Event log:")
    for event in engine.event_log:
        detail = f" {event.detail}" if event.detail else ""
        print(f"  t={event.timestamp:6.2f}  {event.request_id:8s}  {event.event_type.value}{detail}")

    print("\nFinal request states:")
    for req in requests:
        print(f"  {req.request_id}: {req.state.value}")

    assert engine.is_complete()
    assert all(r.state == RequestState.COMPLETED for r in requests)
    print(
        f"\nAll {len(requests)} requests reached COMPLETED. "
        f"Simulated clock stopped at t={engine.clock.now():.2f}."
    )


if __name__ == "__main__":
    main()
