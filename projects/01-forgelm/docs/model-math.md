# Model math

**Status: not yet implemented.** This document will derive the forward-pass
math for every architecture component ForgeLM implements from scratch —
RMSNorm, RoPE, causal self-attention, SwiGLU, and the full decoder block —
alongside the code that implements it, during **Phase 2 — Transformer
correctness** (see `01-forgelm-spec.md` Part 3).

No architecture code exists in this repository yet (`forgelm/model/` is an
empty boundary placeholder as of Phase 0/1 — see `docs/architecture.md`),
so there is nothing to document here yet. This file is committed now only
to keep the repository structure matching the spec's `docs/` layout from
the start.
