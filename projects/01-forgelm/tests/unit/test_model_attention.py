from __future__ import annotations

import pytest
import torch

from forgelm.model import CausalSelfAttention, precompute_rope_angles


def _make_attn_and_angles(
    d_model: int = 16, n_heads: int = 4, seq_len: int = 6
) -> tuple[CausalSelfAttention, torch.Tensor, torch.Tensor]:
    attn = CausalSelfAttention(d_model, n_heads, dropout=0.0)
    cos, sin = precompute_rope_angles(d_model // n_heads, seq_len)
    return attn, cos, sin


def test_attention_output_shape() -> None:
    attn, cos, sin = _make_attn_and_angles(d_model=16, n_heads=4, seq_len=6)
    x = torch.randn(3, 6, 16)
    out = attn(x, cos, sin)
    assert out.shape == (3, 6, 16)


def test_attention_rejects_indivisible_d_model_n_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(d_model=10, n_heads=3)


def test_attention_rejects_mismatched_input_last_dim() -> None:
    attn, cos, sin = _make_attn_and_angles(d_model=16, n_heads=4, seq_len=6)
    x = torch.randn(2, 6, 8)
    with pytest.raises(ValueError, match="16"):
        attn(x, cos, sin)


def test_causal_mask_prevents_future_token_dependence() -> None:
    """spec §1.8: 'causal mask prevents future-token dependence.'

    Changing the tokens strictly *after* position i must not change
    position i's output -- the defining property of causal attention.
    Uses eval mode (dropout=0 anyway, but explicit) so the only source of
    difference between the two forward passes is the input itself.
    """
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 16, 4, 6
    attn, cos, sin = _make_attn_and_angles(d_model, n_heads, seq_len)
    attn.eval()

    x = torch.randn(1, seq_len, d_model)
    x_modified_future = x.clone()
    # Only perturb positions strictly after index 2.
    x_modified_future[:, 3:, :] = torch.randn_like(x_modified_future[:, 3:, :])

    with torch.no_grad():
        out_original = attn(x, cos, sin)
        out_modified = attn(x_modified_future, cos, sin)

    # Positions 0..2 must be identical: they can only attend to
    # themselves and earlier positions, none of which changed.
    torch.testing.assert_close(out_original[:, :3, :], out_modified[:, :3, :], atol=1e-5, rtol=1e-4)
    # Positions 3..5 (the perturbed ones) should generally differ -- a
    # sanity check that the perturbation actually did something and the
    # test isn't vacuously true.
    assert not torch.allclose(out_original[:, 3:, :], out_modified[:, 3:, :], atol=1e-5)


def test_attention_position_zero_only_attends_to_itself() -> None:
    """A second, more direct causal-mask check: token 0's output must be
    invariant to *anything* about tokens 1..T-1, not just to specific
    perturbations of them."""
    torch.manual_seed(1)
    d_model, n_heads, seq_len = 8, 2, 5
    attn, cos, sin = _make_attn_and_angles(d_model, n_heads, seq_len)
    attn.eval()

    x_a = torch.randn(1, seq_len, d_model)
    x_b = x_a.clone()
    x_b[:, 1:, :] = torch.randn(1, seq_len - 1, d_model)  # completely different tail

    with torch.no_grad():
        out_a = attn(x_a, cos, sin)
        out_b = attn(x_b, cos, sin)

    torch.testing.assert_close(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5, rtol=1e-4)
