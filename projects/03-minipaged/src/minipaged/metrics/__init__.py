"""Run-level metrics -- boundary placeholder.

Still not broken out into its own reporting layer. Phase 3's scheduler
(``minipaged.scheduler``) landed ``SchedulerDecision`` (per-tick admitted/
decoded/rejected/cancelled counts, KV block accounting) and Phase 1/2's
KV-memory metrics (``ContiguousMemoryMetrics``, ``PagedKVMetrics``) live
in ``minipaged.kv.contiguous``/``minipaged.kv.manager`` -- all
deliberately colocated with the allocator/scheduler they describe rather
than centralized here. Real request-level latency metrics (queue delay,
TTFT, inter-token latency, throughput -- S1.8) still need a real model
(Phase 5) to measure meaningfully -- a simulated decode tick has no wall-
clock latency to report. This package is where those get assembled and
reported together, once Phase 5-7 give them something real to measure
(Phase 7 benchmark study).
"""

from __future__ import annotations
