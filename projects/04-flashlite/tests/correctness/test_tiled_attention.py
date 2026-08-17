"""Phase 3 exit criterion: 'Tiled implementation correct and benchmarked.'
This file covers the "correct" half, mirroring
tests/correctness/test_naive_attention.py's shape/dtype/causal matrix
exactly (same parametrization, same tolerances) but against
flashlite.ops.attention(variant="tiled") (the V2 CUDA kernel,
src/flashlite/cuda_tiled/) -- so a reader can diff the two test files and
see that V2 is held to literally the same correctness bar as V1, not a
looser one. "Benchmarked" is covered by benchmarks/configs/tiled_comparison.json
+ benchmarks/raw/attention.jsonl (see benchmarks/methodology.md SS9).

Requires a CUDA device; every test here is skipped (not failed) when one is
unavailable, and skipped with a clear reason if the extensions have not
been built yet (`pip install -e .`).
"""

from __future__ import annotations

import pytest
import torch

from flashlite.compare import DEFAULT_ATOL, DEFAULT_RTOL, allclose_compare
from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import DEFAULT_SEED, make_extreme_qkv, make_qkv

# Skipped automatically (with a specific reason) by tests/conftest.py's
# pytest_collection_modifyitems when no CUDA device is present, or when
# either compiled extension hasn't been built yet.
pytestmark = pytest.mark.gpu


def _tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    from flashlite import _cuda_tiled

    return _cuda_tiled.attention_tiled_forward(q, k, v, causal)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])  # includes awkward, non-tile-multiple lengths
@pytest.mark.parametrize("causal", [False, True])
def test_tiled_matches_reference_across_shape_matrix(head_dim: int, seq_len: int, causal: bool) -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=causal)
    actual = _tiled_attention(q, k, v, causal)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("batch,heads", [(1, 1), (4, 1), (1, 8), (3, 5)])
def test_tiled_matches_reference_across_batch_head_counts(batch: int, heads: int) -> None:
    q, k, v = make_qkv(batch=batch, heads=heads, seq_len=64, head_dim=32, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=True)
    actual = _tiled_attention(q, k, v, True)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_tiled_matches_reference_on_extreme_logits() -> None:
    """The tiled kernel's softmax kernel is byte-for-byte V1's
    softmax_rows_kernel (row max subtracted before exponentiating) -- this
    must stay finite and match the reference even at extreme score
    magnitudes, same as V1.
    """
    q, k, v = make_extreme_qkv(
        batch=1, heads=2, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0, device="cuda"
    )
    expected = reference_attention(q, k, v, causal=False)
    actual = _tiled_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_tiled_causal_output_matches_reference_row_zero_identity() -> None:
    """Same behavioral check as V0/V1: row 0 under causal masking must
    exactly reproduce v[..., 0, :] (attends only to itself).
    """
    q, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=32, seed=DEFAULT_SEED, device="cuda")
    out = _tiled_attention(q, k, v, True)
    torch.testing.assert_close(out[0, 0, 0, :], v[0, 0, 0, :], atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)


def test_tiled_is_deterministic_across_repeated_calls() -> None:
    q, k, v = make_qkv(batch=2, heads=2, seq_len=100, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    out1 = _tiled_attention(q, k, v, True)
    out2 = _tiled_attention(q, k, v, True)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_tiled_matches_naive_directly() -> None:
    """V1 and V2 compute the identical mathematical function through two
    different kernel implementations -- they should agree with each other
    (not just each separately with V0) within the same FP32 tolerance,
    since any V1/V2 divergence beyond floating-point reassociation noise
    would mean one of the two tiling loops has an indexing bug the
    V0-comparison tests alone might not isolate as clearly.
    """
    q, k, v = make_qkv(batch=2, heads=4, seq_len=193, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    from flashlite import _cuda_naive, _cuda_tiled

    naive_out = _cuda_naive.attention_naive_forward(q, k, v, True)
    tiled_out = _cuda_tiled.attention_tiled_forward(q, k, v, True)

    result = allclose_compare(tiled_out, naive_out, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_ops_dispatcher_tiled_variant_matches_reference() -> None:
    from flashlite.ops import attention

    q, k, v = make_qkv(batch=1, heads=2, seq_len=48, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    expected = attention(q, k, v, causal=True, variant="reference")
    actual = attention(q, k, v, causal=True, variant="tiled")
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()
