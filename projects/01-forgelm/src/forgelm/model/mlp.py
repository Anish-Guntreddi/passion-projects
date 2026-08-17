"""SwiGLU gated MLP.

Shazeer (2020), "GLU Variants Improve Transformer": in place of a
standard two-layer MLP (``Linear -> activation -> Linear``), SwiGLU uses
three projections and a gated (elementwise-multiplied) hidden state:

    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

``SiLU(z) = z * sigmoid(z)`` (also called "Swish"). This is the MLP used
by LLaMA, Mistral, and most modern open decoder-only LLMs in place of
GELU/ReLU MLPs (FR4: "SwiGLU (or equivalent gated MLP)"). ``F.silu`` is a
plain elementwise activation function -- the same tier of PyTorch
primitive as ``torch.softmax`` or ``nn.Linear`` used elsewhere in this
package, not a "copied model implementation."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUMLP(nn.Module):
    """Gated MLP: ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``, no biases."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
