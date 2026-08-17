"""Phase 1 exit criterion: 'Custom path matches reference within documented
tolerance.' Every case here compares flashlite.ops.attention(variant="naive")
(the V1 CUDA kernel, src/flashlite/cuda_naive/) against
flashlite.reference.attention (V0) on the SAME deterministic input, using
flashlite.compare.allclose_compare with the FP32 tolerances fixed by
ADR 0004 (atol=1e-5, rtol=1e-4).

Requires a CUDA device; every test here is skipped (not failed) when one is
unavailable, and skipped with a clear reason if the extension has not been
built yet (`pip install -e .`).
"""

from __future__ import annotations

import pytest
import torch

from flashlite.compare import DEFAULT_ATOL, DEFAULT_RTOL, allclose_compare
from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import DEFAULT_SEED, make_extreme_qkv, make_qkv

# Skipped automatically (with a specific reason) by tests/conftest.py's
# pytest_collection_modifyitems when no CUDA device is present, or when
# CUDA is present but flashlite._cuda_naive hasn't been built yet.
pytestmark = pytest.mark.gpu


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    from flashlite import _cuda_naive

    return _cuda_naive.attention_naive_forward(q, k, v, causal)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])  # includes awkward, non-tile-multiple lengths
@pytest.mark.parametrize("causal", [False, True])
def test_naive_matches_reference_across_shape_matrix(head_dim: int, seq_len: int, causal: bool) -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=causal)
    actual = _naive_attention(q, k, v, causal)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("batch,heads", [(1, 1), (4, 1), (1, 8), (3, 5)])
def test_naive_matches_reference_across_batch_head_counts(batch: int, heads: int) -> None:
    q, k, v = make_qkv(batch=batch, heads=heads, seq_len=64, head_dim=32, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=True)
    actual = _naive_attention(q, k, v, True)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_naive_matches_reference_on_extreme_logits() -> None:
    """The naive kernel's softmax subtracts the row max too (see
    softmax_rows_kernel in attention_naive.cu) -- this must stay finite and
    match the reference even at extreme score magnitudes.
    """
    q, k, v = make_extreme_qkv(
        batch=1, heads=2, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0, device="cuda"
    )
    expected = reference_attention(q, k, v, causal=False)
    actual = _naive_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_naive_causal_output_matches_reference_row_zero_identity() -> None:
    """Same behavioral check as the V0 reference test, run against the
    actual CUDA kernel: row 0 under causal masking must exactly reproduce
    v[..., 0, :] (attends only to itself).
    """
    q, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=32, seed=DEFAULT_SEED, device="cuda")
    out = _naive_attention(q, k, v, True)
    torch.testing.assert_close(out[0, 0, 0, :], v[0, 0, 0, :], atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)


def test_naive_is_deterministic_across_repeated_calls() -> None:
    q, k, v = make_qkv(batch=2, heads=2, seq_len=100, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    out1 = _naive_attention(q, k, v, True)
    out2 = _naive_attention(q, k, v, True)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_ops_dispatcher_naive_variant_matches_reference() -> None:
    from flashlite.ops import attention

    q, k, v = make_qkv(batch=1, heads=2, seq_len=48, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    expected = attention(q, k, v, causal=True, variant="reference")
    actual = attention(q, k, v, causal=True, variant="naive")
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()
