"""Sampling configuration and (from Phase 5 onward) sampling logic.

Phase 0-1 status: only the data-shape (``SamplingConfig``) exists. Actual
token sampling (temperature/top-k/top-p applied to model logits) has
nothing to sample from until a real model is wired in -- see Phase 5 of
the roadmap and ``docs/decisions/0001-simulation-first-architecture.md``.
"""

from __future__ import annotations

from minipaged.sampling.config import SamplingConfig, SamplingConfigError

__all__ = ["SamplingConfig", "SamplingConfigError"]
