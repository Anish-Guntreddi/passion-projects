"""Causal multi-head self-attention with rotary position embeddings.

Implements the standard decoder-only attention mechanism directly with
tensor reshapes and matmuls -- not ``torch.nn.MultiheadAttention`` or
``torch.nn.functional.scaled_dot_product_attention`` -- so the multi-head
split/merge and causal masking (FR4) stay visible, readable, and this
project's own code, per the "no copied full model implementation"
quality bar (spec §1.7).

Shapes (B = batch, T = sequence length, C = d_model, H = n_heads,
Hd = head_dim = C / H):

    x                (B, T, C)
    qkv_proj(x)      (B, T, 3C)   -> split -> q, k, v, each (B, T, C)
    reshape to heads (B, T, H, Hd) -> transpose -> (B, H, T, Hd)
    apply_rope(q), apply_rope(k)  (rotates each head's Hd-vector; RoPE
                                    operates per-head, not across heads)
    scores = q @ k^T / sqrt(Hd)             (B, H, T, T)
    scores.masked_fill(causal_mask, -inf)   (strictly-upper triangle)
    weights = softmax(scores, dim=-1)
    out = weights @ v                       (B, H, T, Hd)
    merge heads -> (B, T, C) -> out_proj -> (B, T, C)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from forgelm.model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, no biases."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d_model}")

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # each (B, H, T, Hd)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v  # (B, H, T, Hd)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(out))
