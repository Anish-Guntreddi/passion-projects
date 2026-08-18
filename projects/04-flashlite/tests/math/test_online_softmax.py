"""Phase 4 exit criterion (spec SS1.4/roadmap Phase 4): 'Running max +
normalization accumulator implemented/tested independently, then
integrated' -> 'Tests cover extreme score values and multiple tile
boundaries.' This file is that independent test suite: every test here
runs entirely on CPU, with NO dependency on any CUDA kernel or on
`flashlite.ops` -- `src/flashlite/cuda_online_softmax/` (Phase 4's CUDA
integration) does not need to exist yet for this file to pass. See
`docs/online-softmax.md` for the derivation each test below is checking.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashlite.online_softmax import (
    combine_online_stats,
    online_softmax_attention_row,
    online_softmax_row,
    online_softmax_stats,
)
from flashlite.reference.tensors import DEFAULT_SEED, make_extreme_qkv

# --- combine_online_stats (docs/online-softmax.md SS3) ----------------------


def test_combine_online_stats_matches_one_shot_two_element_computation() -> None:
    m, l = combine_online_stats(m_a=1.0, l_a=math.exp(1.0 - 1.0), m_b=3.0, l_b=math.exp(3.0 - 3.0))
    assert m == 3.0
    # Union of {1.0, 3.0}: true max 3.0, l = exp(1-3) + exp(3-3).
    assert math.isclose(l, math.exp(1.0 - 3.0) + 1.0, rel_tol=1e-12)


def test_combine_online_stats_empty_identity_is_a_no_op() -> None:
    m, l = combine_online_stats(m_a=-math.inf, l_a=0.0, m_b=5.0, l_b=2.5)
    assert m == 5.0 and l == 2.5
    m2, l2 = combine_online_stats(m_a=5.0, l_a=2.5, m_b=-math.inf, l_b=0.0)
    assert m2 == 5.0 and l2 == 2.5


def test_combine_online_stats_both_empty_stays_empty() -> None:
    m, l = combine_online_stats(m_a=-math.inf, l_a=0.0, m_b=-math.inf, l_b=0.0)
    assert m == -math.inf and l == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_combine_online_stats_is_associative_and_commutative(seed: int) -> None:
    """Combining three tiles' local stats in any order/grouping must agree
    -- this is the property that makes 'any tiling produces the same
    answer' (SS6's worked example, SS9's chunking-invariance test) true in
    general, not just for the specific groupings those tests happen to try.
    """
    g = torch.Generator().manual_seed(seed)
    tiles = [torch.empty(4).uniform_(-50, 50, generator=g) for _ in range(3)]
    local = []
    for t in tiles:
        tm = float(t.max())
        local.append((tm, float(torch.exp(t - tm).sum())))
    (m1, l1), (m2, l2), (m3, l3) = local

    # (1 combine 2) combine 3
    ma, la = combine_online_stats(m1, l1, m2, l2)
    m_left, l_left = combine_online_stats(ma, la, m3, l3)
    # 1 combine (2 combine 3)
    mb, lb = combine_online_stats(m2, l2, m3, l3)
    m_right, l_right = combine_online_stats(m1, l1, mb, lb)
    # 3 combine 1, then combine 2 (fully different order)
    mc, lc = combine_online_stats(m3, l3, m1, l1)
    m_other, l_other = combine_online_stats(mc, lc, m2, l2)

    assert m_left == m_right == m_other
    assert math.isclose(l_left, l_right, rel_tol=1e-10)
    assert math.isclose(l_left, l_other, rel_tol=1e-10)


# --- online_softmax_stats / online_softmax_row (docs/online-softmax.md SS4, SS6) --


def test_online_softmax_stats_matches_one_shot_computation_worked_example() -> None:
    """The exact worked example in docs/online-softmax.md SS6 -- the true
    row max is deliberately in the MIDDLE tile, not the first, so this
    genuinely exercises a mid-stream running-max update.
    """
    scores = torch.tensor([1.0, 5.0, 3.0, 9.0, 2.0], dtype=torch.float64)
    ref_m = float(scores.max())
    ref_l = float(torch.exp(scores - ref_m).sum())

    for tile_size in [1, 2, 3, 5]:
        m, l = online_softmax_stats(scores, tile_size)
        assert m == ref_m
        assert math.isclose(l, ref_l, rel_tol=1e-12), f"tile_size={tile_size}: l={l} ref_l={ref_l}"


@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])  # matches this repo's non-tile-multiple convention
@pytest.mark.parametrize("tile_size", [1, 3, 8, 32, 100])
def test_online_softmax_row_matches_torch_softmax_across_shape_and_tile_matrix(seq_len: int, tile_size: int) -> None:
    g = torch.Generator().manual_seed(DEFAULT_SEED)
    scores = torch.empty(seq_len, dtype=torch.float64).uniform_(-10, 10, generator=g)
    expected = torch.softmax(scores, dim=-1)
    actual = online_softmax_row(scores, tile_size=tile_size)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_online_softmax_row_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        online_softmax_row(torch.zeros(2, 2), tile_size=1)


def test_online_softmax_row_rejects_non_positive_tile_size() -> None:
    with pytest.raises(ValueError, match="tile_size must be positive"):
        online_softmax_row(torch.zeros(4), tile_size=0)


# --- Extreme logits (docs/online-softmax.md SS4, SS9) ------------------------


def test_online_softmax_row_stays_finite_and_matches_reference_on_extreme_logits() -> None:
    """Scores built the same way make_extreme_qkv builds Q/K (magnitude=80,
    matching every other extreme-logit test in this repo) -- raw scores of
    this magnitude overflow float32's exp() (docs/online-softmax.md SS4),
    which the next test demonstrates directly on this exact input.
    """
    q, k, _v = make_extreme_qkv(batch=1, heads=1, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0)
    scale = 1.0 / math.sqrt(64)
    row = (q[0, 0] @ k[0, 0].T)[0] * scale  # float32, one row of raw QK^T/sqrt(D) scores

    assert float(row.abs().max()) > 90.0, "test fixture must actually exceed float32's ~88.7 exp() overflow point"

    expected = torch.softmax(row, dim=-1)
    actual = online_softmax_row(row, tile_size=7)
    torch.testing.assert_close(actual.to(torch.float32), expected, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(actual).all()


def test_naive_unshifted_exponentiation_actually_overflows_on_the_same_input() -> None:
    """Demonstrates the failure mode docs/online-softmax.md SS4 names: on
    the SAME extreme-logit row the test above handles correctly, naive
    (unshifted, no running-max subtraction) exponentiation overflows to inf
    and its normalized softmax is NaN. This is not asserted from memory --
    it is run here, so a change to make_extreme_qkv's magnitude or this
    repo's tolerance conventions cannot silently make this claim stale
    without a test failing.
    """
    q, k, _v = make_extreme_qkv(batch=1, heads=1, seq_len=64, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0)
    scale = 1.0 / math.sqrt(64)
    row = (q[0, 0] @ k[0, 0].T)[0] * scale  # float32

    naive_exp = torch.exp(row)
    assert torch.isinf(naive_exp).any(), "extreme-logit fixture should overflow float32 exp() when unshifted"
    naive_softmax = naive_exp / naive_exp.sum()
    assert torch.isnan(naive_softmax).any(), "inf / inf should produce NaN in the naive (unstable) computation"

    # The stable reference, and this module's online algorithm, both stay finite on the identical input.
    assert torch.isfinite(torch.softmax(row, dim=-1)).all()
    assert torch.isfinite(online_softmax_row(row, tile_size=5)).all()


# --- online_softmax_attention_row (docs/online-softmax.md SS5, SS7) ----------


@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 257])
@pytest.mark.parametrize("head_dim", [1, 8, 64])
@pytest.mark.parametrize("tile_size", [1, 4, 32])
def test_online_softmax_attention_row_matches_reference_across_shape_matrix(
    seq_len: int, head_dim: int, tile_size: int
) -> None:
    g = torch.Generator().manual_seed(DEFAULT_SEED)
    scores = torch.empty(seq_len, dtype=torch.float64).uniform_(-10, 10, generator=g)
    v_rows = torch.empty(seq_len, head_dim, dtype=torch.float64).uniform_(-1, 1, generator=g)

    expected = torch.softmax(scores, dim=-1) @ v_rows
    actual = online_softmax_attention_row(scores, v_rows, tile_size=tile_size)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("tile_size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("max_position", [0, 4, 6])  # first, middle, last element holds the true max
def test_online_softmax_attention_row_matches_reference_when_max_is_in_a_later_tile(
    tile_size: int, max_position: int
) -> None:
    """docs/online-softmax.md SS7's worked example, generalized: this
    specifically stresses the 'rescale ALREADY-accumulated acc' code path
    -- a bug that only zeroed/rescaled `l` correctly but forgot to apply the
    identical correction to the vector `acc` would pass every test where
    the true max happens to be in the first tile (acc starts at 0, so a
    missing rescale would be invisible) but fail here.
    """
    base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0], dtype=torch.float64)
    scores = base.clone()
    scores[max_position] = 100.0  # far above every other entry, regardless of position
    g = torch.Generator().manual_seed(DEFAULT_SEED)
    v_rows = torch.empty(7, 8, dtype=torch.float64).uniform_(-1, 1, generator=g)

    expected = torch.softmax(scores, dim=-1) @ v_rows
    actual = online_softmax_attention_row(scores, v_rows, tile_size=tile_size)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_online_softmax_attention_row_handles_causal_style_masked_tiles() -> None:
    """Rows containing -inf entries (docs/attention-math.md SS3's causal
    convention), including a tile that is ENTIRELY -inf (SS5's 'empty
    identity' case) -- must be skipped as a no-op, not produce NaN.
    """
    scores = torch.tensor([1.0, 2.0, -math.inf, -math.inf, -math.inf, 3.0], dtype=torch.float64)
    g = torch.Generator().manual_seed(DEFAULT_SEED)
    v_rows = torch.empty(6, 4, dtype=torch.float64).uniform_(-1, 1, generator=g)

    expected = torch.softmax(scores, dim=-1) @ v_rows
    actual = online_softmax_attention_row(scores, v_rows, tile_size=2)  # tile [-inf, -inf] is entirely masked
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    assert torch.isfinite(actual).all()


def test_online_softmax_attention_row_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"\[S, D\]"):
        online_softmax_attention_row(torch.zeros(4), torch.zeros(5, 3), tile_size=1)


def test_online_softmax_attention_row_rejects_fully_masked_row() -> None:
    scores = torch.full((4,), -math.inf, dtype=torch.float64)
    v_rows = torch.zeros(4, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="fully masked"):
        online_softmax_attention_row(scores, v_rows, tile_size=2)
