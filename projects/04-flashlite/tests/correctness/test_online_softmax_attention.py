"""Phase 4 exit criterion (roadmap): 'Tests cover extreme score values and
multiple tile boundaries' -- this file is the CUDA-INTEGRATION half of that
(tests/math/test_online_softmax.py is the CPU-only, kernel-independent
half; docs/online-softmax.md SS9 explains the split). Mirrors
tests/correctness/test_tiled_attention.py's shape/dtype/causal matrix
exactly (same parametrization, same tolerances) against
flashlite.ops.attention(variant="online_softmax") (the V3 CUDA kernel,
src/flashlite/cuda_online_softmax/), so a reader can diff the two files and
see V3 is held to literally the same correctness bar as V2 -- plus one
V3-specific test that exercises MULTIPLE internal 256-element "wave"
boundaries within a single row (kOnlineSoftmaxBlockSize=256,
attention_online_softmax.cuh), with the true row maximum deliberately
placed in a LATE wave, which the smaller seq_len values already covered
above cannot exercise (S <= 257 never crosses a second 256-wide wave more
than barely).

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


def _online_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    from flashlite import _cuda_online_softmax

    return _cuda_online_softmax.attention_online_softmax_forward(q, k, v, causal)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])  # includes awkward, non-tile-multiple lengths
@pytest.mark.parametrize("causal", [False, True])
def test_online_softmax_matches_reference_across_shape_matrix(head_dim: int, seq_len: int, causal: bool) -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=causal)
    actual = _online_softmax_attention(q, k, v, causal)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("batch,heads", [(1, 1), (4, 1), (1, 8), (3, 5)])
def test_online_softmax_matches_reference_across_batch_head_counts(batch: int, heads: int) -> None:
    q, k, v = make_qkv(batch=batch, heads=heads, seq_len=64, head_dim=32, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=True)
    actual = _online_softmax_attention(q, k, v, True)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_online_softmax_matches_reference_on_extreme_logits() -> None:
    """docs/online-softmax.md SS4/SS9: online_softmax_rows_kernel's running
    max/sum combine must stay finite and match the reference even at
    extreme score magnitudes, same fixture every other extreme-logit test
    in this repo uses.
    """
    q, k, v = make_extreme_qkv(
        batch=1, heads=2, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0, device="cuda"
    )
    expected = reference_attention(q, k, v, causal=False)
    actual = _online_softmax_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("seq_len", [300, 600, 1000])  # > kOnlineSoftmaxBlockSize=256: multiple internal waves
def test_online_softmax_correct_when_true_max_is_in_a_late_wave(seq_len: int) -> None:
    """Exercises docs/online-softmax.md's Phase-A/Phase-B combine at the
    CUDA level with MULTIPLE 256-element 'wave' boundaries within a single
    row (kOnlineSoftmaxBlockSize=256, attention_online_softmax.cuh) -- the
    smaller shapes in the shape-matrix test above never cross more than one
    such boundary. Deliberately places one key position's raw score far
    above every other position's, LATE in the row (past several full
    waves), so a kernel that only correctly handled a max discovered in
    wave 0 (a plausible off-by-one: initializing `m` from thread 0's first
    element instead of the -FLT_MAX sentinel) would fail here but pass the
    shape-matrix test above.
    """
    q, k, v = make_qkv(batch=1, heads=2, seq_len=seq_len, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    late_key = seq_len - 7  # near the end, comfortably past several 256-element waves
    k[:, :, late_key, :] = 50.0  # large, uniform K row -> a large dot product for every query that can see it
    q[:, :, :, :] = 1.0  # keeps every row's score for late_key large and consistent, not sign-cancelling

    expected = reference_attention(q, k, v, causal=False)
    actual = _online_softmax_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()
    # Every row now overwhelmingly attends to late_key -- confirms the test
    # fixture actually stresses the intended code path, not just a shape
    # that happens to also pass.
    torch.testing.assert_close(actual[0, 0], v[0, 0, late_key, :].expand_as(actual[0, 0]), atol=1e-3, rtol=1e-3)


def test_online_softmax_causal_output_matches_reference_row_zero_identity() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=32, seed=DEFAULT_SEED, device="cuda")
    out = _online_softmax_attention(q, k, v, True)
    torch.testing.assert_close(out[0, 0, 0, :], v[0, 0, 0, :], atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)


def test_online_softmax_is_deterministic_across_repeated_calls() -> None:
    q, k, v = make_qkv(batch=2, heads=2, seq_len=100, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    out1 = _online_softmax_attention(q, k, v, True)
    out2 = _online_softmax_attention(q, k, v, True)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_online_softmax_matches_tiled_directly() -> None:
    """V2 and V3 compute the identical mathematical function through two
    different softmax algorithms (two-pass vs combined-pass online) -- they
    should agree with each other, not just each separately with V0.
    """
    q, k, v = make_qkv(batch=2, heads=4, seq_len=193, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    from flashlite import _cuda_online_softmax, _cuda_tiled

    tiled_out = _cuda_tiled.attention_tiled_forward(q, k, v, True)
    online_out = _cuda_online_softmax.attention_online_softmax_forward(q, k, v, True)

    result = allclose_compare(online_out, tiled_out, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_ops_dispatcher_online_softmax_variant_matches_reference() -> None:
    from flashlite.ops import attention

    q, k, v = make_qkv(batch=1, heads=2, seq_len=48, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    expected = attention(q, k, v, causal=True, variant="reference")
    actual = attention(q, k, v, causal=True, variant="online_softmax")
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()
