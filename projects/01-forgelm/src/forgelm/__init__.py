"""ForgeLM: a from-scratch decoder-only Transformer training stack.

This package is organized by strict component boundaries (see
``docs/architecture.md``):

- ``forgelm.tokenizer``  -- text <-> token id conversion only.
- ``forgelm.dataset``    -- reading text, building token arrays, splits, batching.
- ``forgelm.model``      -- architecture modules only (no optimization logic).
- ``forgelm.training``   -- optimizer, schedules, precision, checkpointing.
- ``forgelm.evaluation`` -- loss/perplexity/generation evaluation.
- ``forgelm.generation`` -- sampling from a trained checkpoint.
- ``forgelm.benchmarks`` -- throughput/memory experiments (never training math).
- ``forgelm.cli``        -- composes the above; must not contain model logic.

Phase 0 and Phase 1 of the project roadmap are implemented as of this
version: repo foundation (config, seeding, smoke test) and tokenization +
dataset construction. See ``docs/decisions/`` for the ADRs backing every
open design decision, and the top-level spec (``01-forgelm-spec.md`` in the
portfolio repo root) for the full roadmap.
"""

__version__ = "0.1.0"
