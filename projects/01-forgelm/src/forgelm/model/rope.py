"""Rotary position embeddings (RoPE).

Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position
Embedding": instead of adding a position vector to token embeddings,
RoPE rotates each (query, key) vector in ``head_dim / 2`` independent 2D
sub-planes by an angle proportional to its sequence position, so the dot
product between a rotated query at position ``i`` and a rotated key at
position ``j`` depends only on the relative offset ``i - j`` and the two
vectors' original content -- relative position is encoded directly into
the attention score, with no additive position embedding anywhere. This
is the positional scheme used by LLaMA, Mistral, GPT-NeoX and most
modern open decoder-only LLMs.

This module implements the "rotate-half" formulation (splits the head
dimension into two halves, rather than interleaving adjacent pairs as
the original paper does -- mathematically equivalent, just a different
choice of which coordinate pairs get rotated together; this is the
formulation LLaMA and most modern implementations use):

    freq_k      = theta ** (-2k / head_dim),   k = 0 .. head_dim/2 - 1
    angle(p, k) = p * freq_k                    (p = sequence position)

    rotate_half([x1, x2]) = [-x2, x1]           (x1, x2: first/second half)
    RoPE(x, p) = x * cos(angle(p)) + rotate_half(x) * sin(angle(p))

``cos``/``sin`` are precomputed once per ``(head_dim, max_seq_len,
theta)`` and each duplicated across both halves so they broadcast
directly against the full ``head_dim`` vector.
"""

from __future__ import annotations

import torch


def precompute_rope_angles(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``cos``/``sin`` tables, each shape ``(max_seq_len, head_dim)``.

    ``head_dim`` must be even -- RoPE rotates ``head_dim / 2`` independent
    2D planes. Pure function of its arguments: no RNG, so it is
    deterministic and safe to cache.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    half = head_dim // 2
    exponents = torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / head_dim
    freqs = 1.0 / (theta**exponents)  # (half,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len,)
    angles = torch.outer(positions, freqs)  # (max_seq_len, half)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # Duplicate across both halves so the table's last dim is head_dim and
    # broadcasts directly against (..., head_dim) tensors in apply_rope.
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim of ``x`` in half and return ``cat([-x2, x1])``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``x``.

    Args:
        x: shape ``(..., seq_len, head_dim)``.
        cos, sin: shape ``(seq_len, head_dim)`` -- e.g. a slice of
            :func:`precompute_rope_angles`'s output for the positions
            present in ``x``. Broadcasts against any number of leading
            dims (batch, heads, ...).
    """
    if x.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"head_dim mismatch: x has {x.shape[-1]}, cos/sin table has {cos.shape[-1]}"
        )
    if x.shape[-2] != cos.shape[-2]:
        raise ValueError(
            f"seq_len mismatch: x has {x.shape[-2]}, cos/sin table has {cos.shape[-2]}"
        )
    return x * cos + rotate_half(x) * sin
