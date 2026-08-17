"""Run-level metrics -- boundary placeholder.

Not implemented yet for scheduler-level metrics (queue delay, TTFT,
inter-token latency, throughput -- S1.8), which need a real scheduler
(Phase 3) to measure. Phase 1's KV-memory metrics
(``ContiguousMemoryMetrics``) currently live in
``minipaged.kv.contiguous`` instead of here, deliberately: they are
intrinsic to the allocator being measured, not to a scheduling policy.
They may be re-exported from this package once request-level and
KV-level metrics need to be reported together (Phase 7 benchmark study).
"""

from __future__ import annotations
