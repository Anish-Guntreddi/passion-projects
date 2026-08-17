"""One pre-norm Transformer decoder block.

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Pre-norm (normalize *before* each sub-layer, rather than after it and the
residual add) is used because it is what every modern decoder-only LLM
(GPT-2 onward, LLaMA, ...) uses -- it keeps gradients well-behaved
through deep stacks without needing the delicately-tuned learning-rate
warmup the original post-norm Transformer required.
"""

from __future__ import annotations

import torch
from torch import nn

from forgelm.model.attention import CausalSelfAttention
from forgelm.model.mlp import SwiGLUMLP
from forgelm.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """RMSNorm -> causal self-attention -> residual, RMSNorm -> SwiGLU -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.mlp_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = SwiGLUMLP(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x
