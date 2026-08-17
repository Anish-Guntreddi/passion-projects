"""MiniPaged: an educational LLM serving runtime.

Demonstrates the systems ideas behind modern high-throughput LLM serving
(paged KV-cache allocation, continuous batching, admission control,
prefill/decode scheduling) as a small, interview-explainable codebase --
not a clone of a production runtime. See ``03-minipaged-spec.md`` in the
portfolio repo root for the full PRD, tech stack plan, and roadmap.

Implementation strategy (spec S1.6, "load-bearing"): built in layers. The
domain simulator (Phase 0) and contiguous KV baseline (Phase 1) run with
NO real model -- KV allocation is simulated as abstract "token slots"
independent of any actual tensor. Only after the allocator and scheduler
invariants are well-tested (Phases 2-4) does a real model execution path
get connected (Phase 5). See ``docs/decisions/0001-simulation-first-architecture.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"
