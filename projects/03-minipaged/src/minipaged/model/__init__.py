"""Real model adapter -- boundary placeholder.

Not implemented yet. Phase 5 of the roadmap integrates one small
Hugging Face/PyTorch causal LM (FR1) behind an interface the scheduler and
KV subsystem stay independent of, per the spec's load-bearing
implementation strategy (S1.6) and
``docs/decisions/0001-simulation-first-architecture.md``. Agent execution
rule #1: no real model is integrated before allocator/scheduler tests
exist (Phases 0-4 gate Phase 5).
"""

from __future__ import annotations
