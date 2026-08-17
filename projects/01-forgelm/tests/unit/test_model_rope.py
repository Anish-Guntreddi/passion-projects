from __future__ import annotations

import pytest
import torch

from forgelm.model import apply_rope, precompute_rope_angles

# -- precompute_rope_angles ------------------------------------------------------


def test_precompute_rope_angles_shape() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=16)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_precompute_rope_angles_is_deterministic() -> None:
    cos_a, sin_a = precompute_rope_angles(head_dim=8, max_seq_len=16, theta=10000.0)
    cos_b, sin_b = precompute_rope_angles(head_dim=8, max_seq_len=16, theta=10000.0)
    assert torch.equal(cos_a, cos_b)
    assert torch.equal(sin_a, sin_b)


def test_precompute_rope_angles_position_zero_has_zero_angle() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=4)
    torch.testing.assert_close(cos[0], torch.ones(8))
    torch.testing.assert_close(sin[0], torch.zeros(8))


@pytest.mark.parametrize("head_dim", [0, -2, 3, 7])
def test_precompute_rope_angles_rejects_bad_head_dim(head_dim: int) -> None:
    with pytest.raises(ValueError):
        precompute_rope_angles(head_dim=head_dim, max_seq_len=4)


def test_precompute_rope_angles_rejects_non_positive_max_seq_len() -> None:
    with pytest.raises(ValueError):
        precompute_rope_angles(head_dim=8, max_seq_len=0)


# -- apply_rope: shape / determinism ----------------------------------------------


def test_apply_rope_preserves_shape() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 16, 8)  # (batch, heads, seq, head_dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_apply_rope_is_deterministic() -> None:
    torch.manual_seed(0)
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 16, 8)
    out_a = apply_rope(x, cos, sin)
    out_b = apply_rope(x, cos, sin)
    assert torch.equal(out_a, out_b)


def test_apply_rope_rejects_head_dim_mismatch() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 16, 4)
    with pytest.raises(ValueError, match="head_dim"):
        apply_rope(x, cos, sin)


def test_apply_rope_rejects_seq_len_mismatch() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 10, 8)
    with pytest.raises(ValueError, match="seq_len"):
        apply_rope(x, cos, sin)


# -- correctness properties --------------------------------------------------------


def test_apply_rope_position_zero_is_identity() -> None:
    cos, sin = precompute_rope_angles(head_dim=8, max_seq_len=4)
    x = torch.randn(1, 8)
    out = apply_rope(x.unsqueeze(0), cos[:1], sin[:1]).squeeze(0)
    torch.testing.assert_close(out, x)


def test_apply_rope_preserves_vector_norm() -> None:
    """RoPE rotates each 2D sub-plane by an angle -- an orthogonal
    transform, which by definition preserves vector norm."""
    torch.manual_seed(2)
    head_dim = 16
    cos, sin = precompute_rope_angles(head_dim=head_dim, max_seq_len=32)
    x = torch.randn(5, head_dim)
    out = apply_rope(x.unsqueeze(0), cos[:5], sin[:5]).squeeze(0)
    torch.testing.assert_close(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_apply_rope_relative_position_property() -> None:
    """The core RoPE correctness property (Su et al. 2021): the dot
    product of a rotated query at position i and a rotated key at
    position j depends only on the relative offset (i - j), not on i and
    j individually. Reference/cross-check test on tiny tensors (spec
    §1.8)."""
    torch.manual_seed(3)
    head_dim = 8
    max_len = 64
    cos, sin = precompute_rope_angles(head_dim=head_dim, max_seq_len=max_len)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    def rotated_dot(pos_q: int, pos_k: int) -> torch.Tensor:
        rq = apply_rope(q.unsqueeze(0), cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1]).squeeze(0)
        rk = apply_rope(k.unsqueeze(0), cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1]).squeeze(0)
        return torch.dot(rq, rk)

    offset_3_a = rotated_dot(5, 2)
    offset_3_b = rotated_dot(40, 37)
    offset_0 = rotated_dot(10, 10)

    torch.testing.assert_close(offset_3_a, offset_3_b, atol=1e-4, rtol=1e-4)
    # Sanity: a *different* relative offset should (generically, for
    # random q/k) give a measurably different dot product.
    assert (offset_3_a - offset_0).abs().item() > 1e-3
