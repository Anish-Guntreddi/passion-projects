from __future__ import annotations

import pytest
import torch

from forgelm.model import RMSNorm

# -- reference cross-check ------------------------------------------------------
#
# Independently recomputes the RMSNorm formula with different tensor ops
# than the implementation uses (sum()/n instead of mean(), sqrt(1/x)
# instead of rsqrt()), so this is a genuine cross-check against the
# documented formula, not just re-running the same code (spec §1.8:
# "RMSNorm vs reference formula").


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    n = x.shape[-1]
    mean_sq = x.pow(2).sum(dim=-1, keepdim=True) / n
    rms = torch.sqrt(mean_sq + eps)
    return (x / rms) * weight


def test_rmsnorm_matches_reference_formula() -> None:
    torch.manual_seed(0)
    d_model = 16
    norm = RMSNorm(d_model, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(d_model) + 0.5)  # non-trivial gain
    x = torch.randn(4, 7, d_model) * 3.0

    actual = norm(x)
    expected = _reference_rmsnorm(x, norm.weight.detach(), norm.eps)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_rmsnorm_output_shape_matches_input() -> None:
    norm = RMSNorm(32)
    x = torch.randn(2, 5, 32)
    assert norm(x).shape == x.shape


def test_rmsnorm_is_scale_invariant_up_to_gain() -> None:
    """RMSNorm(c * x) == RMSNorm(x) for any c > 0, since rescaling x also
    rescales its own RMS by the same factor -- the defining property of
    RMS normalization (eps aside)."""
    torch.manual_seed(1)
    norm = RMSNorm(8, eps=0.0)  # eps=0 makes the invariance exact
    x = torch.randn(3, 8) + 0.1  # avoid the degenerate all-zero vector
    out_a = norm(x)
    out_b = norm(x * 5.0)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-4)


def test_rmsnorm_zero_input_is_finite_via_eps() -> None:
    norm = RMSNorm(4, eps=1e-6)
    out = norm(torch.zeros(1, 4))
    assert torch.isfinite(out).all()


def test_rmsnorm_rejects_mismatched_last_dim() -> None:
    norm = RMSNorm(16)
    with pytest.raises(ValueError, match="16"):
        norm(torch.randn(2, 8))
