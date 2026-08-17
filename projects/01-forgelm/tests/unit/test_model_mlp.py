from __future__ import annotations

import torch

from forgelm.model import SwiGLUMLP


def test_swiglu_output_shape() -> None:
    mlp = SwiGLUMLP(d_model=16, d_ff=32, dropout=0.0)
    x = torch.randn(3, 5, 16)
    assert mlp(x).shape == (3, 5, 16)


def test_swiglu_matches_manual_silu_gate_formula() -> None:
    """Reference cross-check: recompute SwiGLU by hand (z * sigmoid(z) for
    the SiLU gate, via torch.sigmoid rather than F.silu) using the
    module's own learned weights, and compare against the module's
    forward output."""
    torch.manual_seed(0)
    mlp = SwiGLUMLP(d_model=8, d_ff=12, dropout=0.0)
    mlp.eval()
    x = torch.randn(4, 8)

    with torch.no_grad():
        gate_pre = x @ mlp.gate_proj.weight.T
        up = x @ mlp.up_proj.weight.T
        silu_manual = gate_pre * torch.sigmoid(gate_pre)
        expected = (silu_manual * up) @ mlp.down_proj.weight.T
        actual = mlp(x)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_swiglu_zero_input_is_zero_output() -> None:
    # gate_pre = 0 -> silu(0) = 0 -> gated product = 0 regardless of up_proj,
    # so the whole MLP maps the zero vector to zero (no biases anywhere).
    mlp = SwiGLUMLP(d_model=8, d_ff=16, dropout=0.0)
    mlp.eval()
    out = mlp(torch.zeros(1, 8))
    torch.testing.assert_close(out, torch.zeros(1, 8), atol=1e-6, rtol=1e-6)


def test_swiglu_dropout_zero_is_deterministic() -> None:
    mlp = SwiGLUMLP(d_model=8, d_ff=16, dropout=0.0)
    mlp.eval()
    x = torch.randn(2, 8)
    out_a = mlp(x)
    out_b = mlp(x)
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)
