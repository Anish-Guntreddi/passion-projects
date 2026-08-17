"""Decoder-only Transformer architecture (RMSNorm, RoPE, attention, SwiGLU MLP).

Not yet implemented -- this is Phase 2 of the roadmap (see
``01-forgelm-spec.md`` Part 3). This package exists now so the module
boundary is fixed from the start: ``forgelm.model`` will only ever contain
architecture code, never optimizer/training/checkpointing logic.
"""
