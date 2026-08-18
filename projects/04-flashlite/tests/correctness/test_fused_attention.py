"""Phase 5 exit criterion (roadmap): 'Peak-memory scales as designed;
output matches reference.' This file covers BOTH halves against
flashlite.ops.attention(variant="fused") (the V4 CUDA kernel,
src/flashlite/cuda_fused/): the "output matches reference" half mirrors
tests/correctness/test_tiled_attention.py's shape/dtype/causal matrix
exactly (same parametrization, same tolerances), and
test_fused_peak_memory_grows_far_slower_than_materializing_variants below
is the "peak-memory scales as designed" half, MEASURED via
torch.cuda.max_memory_allocated() from actual kernel calls -- not merely
asserted from the kernel's design (attention_fused.cuh's header already
argues this from first principles; this test is the empirical check that
argument predicts).

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


def _fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    from flashlite import _cuda_fused

    return _cuda_fused.attention_fused_forward(q, k, v, causal)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])  # includes awkward, non-tile-multiple lengths
@pytest.mark.parametrize("causal", [False, True])
def test_fused_matches_reference_across_shape_matrix(head_dim: int, seq_len: int, causal: bool) -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=causal)
    actual = _fused_attention(q, k, v, causal)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("batch,heads", [(1, 1), (4, 1), (1, 8), (3, 5)])
def test_fused_matches_reference_across_batch_head_counts(batch: int, heads: int) -> None:
    q, k, v = make_qkv(batch=batch, heads=heads, seq_len=64, head_dim=32, seed=DEFAULT_SEED, device="cuda")

    expected = reference_attention(q, k, v, causal=True)
    actual = _fused_attention(q, k, v, True)

    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_fused_matches_reference_on_extreme_logits() -> None:
    """The fused kernel's online update (attention_fused.cu) is the same
    combine identity docs/online-softmax.md derives -- must stay finite and
    match the reference even at extreme score magnitudes, same fixture
    every other extreme-logit test in this repo uses.
    """
    q, k, v = make_extreme_qkv(
        batch=1, heads=2, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0, device="cuda"
    )
    expected = reference_attention(q, k, v, causal=False)
    actual = _fused_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


@pytest.mark.parametrize("seq_len", [300, 600, 1000])  # > kFusedTileDim=32: multiple streamed KV tiles
def test_fused_correct_when_true_max_is_in_a_late_kv_tile(seq_len: int) -> None:
    """Mirrors test_online_softmax_attention.py's equivalent test: places a
    key position with a far-larger score LATE in the sequence (past many
    32-row KV tiles the fused kernel streams through sequentially), so a
    kernel that only correctly rescaled `acc`/`l` for the FIRST tile's max
    (missing the `acc[d] = acc[d] * correction + ...` rescale for a later
    tile) would fail here.
    """
    q, k, v = make_qkv(batch=1, heads=2, seq_len=seq_len, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    late_key = seq_len - 5
    k[:, :, late_key, :] = 50.0
    q[:, :, :, :] = 1.0

    expected = reference_attention(q, k, v, causal=False)
    actual = _fused_attention(q, k, v, False)

    assert torch.isfinite(actual).all()
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()
    torch.testing.assert_close(actual[0, 0], v[0, 0, late_key, :].expand_as(actual[0, 0]), atol=1e-3, rtol=1e-3)


def test_fused_causal_output_matches_reference_row_zero_identity() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=32, seed=DEFAULT_SEED, device="cuda")
    out = _fused_attention(q, k, v, True)
    torch.testing.assert_close(out[0, 0, 0, :], v[0, 0, 0, :], atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)


def test_fused_is_deterministic_across_repeated_calls() -> None:
    q, k, v = make_qkv(batch=2, heads=2, seq_len=100, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    out1 = _fused_attention(q, k, v, True)
    out2 = _fused_attention(q, k, v, True)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_fused_matches_online_softmax_directly() -> None:
    """V3 and V4 compute the identical mathematical function -- V3 via a
    materialized-then-online-normalized score matrix, V4 via the same
    online recurrence applied with no score matrix at all -- they should
    agree with each other, not just each separately with V0.
    """
    q, k, v = make_qkv(batch=2, heads=4, seq_len=193, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    from flashlite import _cuda_fused, _cuda_online_softmax

    online_out = _cuda_online_softmax.attention_online_softmax_forward(q, k, v, True)
    fused_out = _cuda_fused.attention_fused_forward(q, k, v, True)

    result = allclose_compare(fused_out, online_out, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


def test_ops_dispatcher_fused_variant_matches_reference() -> None:
    from flashlite.ops import attention

    q, k, v = make_qkv(batch=1, heads=2, seq_len=48, head_dim=64, seed=DEFAULT_SEED, device="cuda")
    expected = attention(q, k, v, causal=True, variant="reference")
    actual = attention(q, k, v, causal=True, variant="fused")
    result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    assert result.passed, result.summary()


# --- Phase 5's namesake exit criterion: measured peak-memory scaling -------


def test_fused_peak_memory_grows_far_slower_than_materializing_variants() -> None:
    """attention_fused.cuh's header argues from the kernel's own design
    that V4 allocates no [B, H, S, S] scores buffer at all, so its peak
    memory should scale roughly with the irreducible O(S*D) output/input
    size, while every earlier variant (checked here via V3, which shares
    V4's online-softmax math exactly, so this isolates the ONE variable
    that differs -- materialization -- from any softmax-algorithm
    difference) allocates an O(S^2) scores buffer and should show
    materially faster growth as S increases.

    This measures torch.cuda.max_memory_allocated() around one forward call
    per (variant, seq_len) point -- not a claim from the design doc alone.
    The assertion is on the TREND (the v3-to-v4 peak-memory ratio must
    widen, substantially, as seq_len grows), not on any single hard-coded
    byte count, so it is robust to PyTorch caching-allocator overhead/
    rounding while still being a concrete, falsifiable check of "peak-memory
    scales as designed" (roadmap Phase 5's exit criterion).
    """
    from flashlite import _cuda_fused, _cuda_online_softmax

    seq_lens = [512, 1024, 2048, 4096]
    batch, heads, head_dim = 1, 8, 64

    peak_fused = []
    peak_materialized = []
    for seq_len in seq_lens:
        q, k, v = make_qkv(batch=batch, heads=heads, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED, device="cuda")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _cuda_fused.attention_fused_forward(q, k, v, True)
        torch.cuda.synchronize()
        peak_fused.append(torch.cuda.max_memory_allocated())

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _cuda_online_softmax.attention_online_softmax_forward(q, k, v, True)
        torch.cuda.synchronize()
        peak_materialized.append(torch.cuda.max_memory_allocated())

        del q, k, v
        torch.cuda.empty_cache()

    ratios = [peak_materialized[i] / peak_fused[i] for i in range(len(seq_lens))]

    # The gap between "materializes [S,S]" and "never materializes it"
    # must widen as S grows -- this is the O(S^2) vs O(S*D) prediction,
    # stated as a monotonic trend rather than a fragile exact number.
    assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1)), (
        f"expected the materialized/fused peak-memory ratio to grow monotonically with seq_len, got {ratios} "
        f"for seq_lens={seq_lens} (fused={peak_fused}, materialized={peak_materialized})"
    )
    # And substantially, not by noise alone, across the full sweep.
    assert ratios[-1] > 2.0 * ratios[0], (
        f"expected the ratio at the largest seq_len to be well over 2x the ratio at the smallest; got "
        f"ratios={ratios} for seq_lens={seq_lens}"
    )
