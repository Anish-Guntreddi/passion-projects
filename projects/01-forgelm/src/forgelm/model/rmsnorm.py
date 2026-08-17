"""RMSNorm (root-mean-square layer normalization).

Zhang & Sennrich (2019), "Root Mean Square Layer Normalization": unlike
LayerNorm, RMSNorm does not re-center the input (no mean subtraction) --
it only rescales by the root-mean-square of the activations, then applies
a learned per-channel gain. This is what every modern open decoder-only
LLM (LLaMA, Mistral, Gemma, ...) uses in place of LayerNorm; ForgeLM
reimplements it from the formula rather than importing it from a
library, per the "no copied full model implementation" quality bar
(spec §1.7).

Formula, elementwise over the last dimension of ``x`` (shape
``(..., d_model)``):

    rms(x)      = sqrt(mean(x_i^2) + eps)
    RMSNorm(x)  = (x / rms(x)) * weight

``weight`` (shape ``(d_model,)``) is a learned per-channel gain,
initialized to ones, so training starts as an identity rescale.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension of the input."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"RMSNorm configured for last dim {self.d_model}, got input with "
                f"last dim {x.shape[-1]}"
            )
        # Normalize in float32 regardless of the input dtype: the
        # mean-of-squares reduction loses precision fast under fp16/bf16,
        # and this is cheap relative to the matmuls surrounding it in a
        # TransformerBlock (standard practice -- e.g. LLaMA's RMSNorm).
        in_dtype = x.dtype
        x_fp32 = x.float()
        mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(mean_sq + self.eps)
        return (normed * self.weight.float()).to(in_dtype)
