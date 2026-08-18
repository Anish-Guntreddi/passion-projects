"""V3's online-softmax "unit" (spec SS1.4 hard requirement): the running
max (m) / running normalization sum (l) / running weighted-output
accumulator (acc) update rule, implemented and tested independently of any
CUDA kernel or attention integration -- in plain, deliberately-unvectorized
Python/PyTorch loops so each step matches docs/online-softmax.md's
derivation line-for-line, rather than hiding the algorithm behind a
vectorized shortcut or (per the spec's hard constraint 1: "never copy the
FlashAttention production implementation wholesale") any existing library
implementation.

docs/online-softmax.md derives the math this module transcribes; read that
file first. tests/math/test_online_softmax.py is what "tested independently
before integration into attention" (spec SS1.4) means concretely: every
function here is checked against torch.softmax / a from-scratch reference
with NO dependency on any CUDA kernel, before src/flashlite/cuda_online_softmax/
(V3's CUDA kernel, which performs the identical combine rule -- per-thread,
then per-block, instead of per-Python-loop-iteration) or
src/flashlite/cuda_fused/ (V4, the same per-element recurrence again, this
time streamed directly from QK^T with no scores buffer at all) exist.
"""

from __future__ import annotations

import math

import torch

# Sentinel for "no data folded in yet" (the identity element for
# combine_online_stats). math.inf, not a large-but-finite float, because
# this module works in Python float / float64 torch tensors where IEEE-754
# -inf arithmetic is safe and exact (exp(-inf) == 0.0, no NaN) as long as
# the (m == -math.inf) special case below is handled explicitly wherever
# THIS OWN OLD state, not an operand's magnitude, could itself be -inf.
# (The CUDA kernel in cuda_online_softmax/ uses -FLT_MAX instead, matching
# this repo's existing softmax_rows_kernel float32 convention -- documented
# there, not here, since it is a float32-hardware-specific choice.)
_NEG_INF = -math.inf


def combine_online_stats(m_a: float, l_a: float, m_b: float, l_b: float) -> tuple[float, float]:
    """Combines two partial (running max, running normalization sum) pairs
    -- each computed independently over its own disjoint subset of a row's
    raw scores -- into the single (m, l) pair that subset's UNION would
    have produced, without revisiting any individual element of either
    subset.

        m = max(m_a, m_b)
        l = l_a * exp(m_a - m) + l_b * exp(m_b - m)

    This is the one identity the entire online-softmax algorithm rests on
    (docs/online-softmax.md SS3): each term `l_x * exp(m_x - m)`
    re-expresses subset x's contribution relative to the NEW combined max
    m, using the fact that `exp(v - m_x) * exp(m_x - m) == exp(v - m)` for
    every original element v that subset x's l_x was built from -- so a
    SINGLE scalar correction factor, exp(m_x - m), suffices to correct
    subset x's *entire* accumulated contribution simultaneously, however
    many elements it contains. This is what "how prior partial outputs are
    rescaled when a new maximum appears" (spec SS1.4) means at the level of
    a single scalar statistic; `online_softmax_attention_row` below applies
    the identical correction factor to a whole output vector, not just a
    scalar.

    Associative and commutative: combining tiles in any order, or with any
    grouping, produces the same final (m, l) -- see
    tests/math/test_online_softmax.py::test_combine_online_stats_is_associative_and_commutative.

    An empty/not-yet-seen subset is represented by (m=-inf, l=0) (the
    identity element `online_softmax_stats`'s loop starts from, and what a
    fully causally-masked tile's local (max, sum) collapses to). Combining
    it with anything must be a no-op; the general formula alone cannot
    express that when BOTH inputs are the empty identity simultaneously
    (`exp(-inf - -inf)` is `exp(nan)`, i.e. `nan`, not `exp(0)=1`), so that
    one case is special-cased explicitly below rather than left to float
    arithmetic to resolve.
    """
    m = max(m_a, m_b)
    if m == _NEG_INF:
        # Both subsets are the (-inf, 0) empty identity.
        return _NEG_INF, 0.0
    ca = 0.0 if m_a == _NEG_INF else math.exp(m_a - m)
    cb = 0.0 if m_b == _NEG_INF else math.exp(m_b - m)
    return m, l_a * ca + l_b * cb


def online_softmax_stats(scores: torch.Tensor, tile_size: int) -> tuple[float, float]:
    """Computes `(m, l) = (max(scores), sum(exp(scores - max(scores))))` --
    the two numbers full-row softmax normalization needs -- by walking
    `scores` (a 1-D tensor: one query row's raw, pre-scale-applied,
    pre-softmax logits) in sequential chunks ("tiles") of `tile_size`
    elements (the last tile may be shorter), maintaining a running (m, l)
    pair via `combine_online_stats` instead of ever looking at the whole
    row in one pass the way `torch.softmax` (and V0/V1/V2's two-pass
    `softmax_rows_kernel`) does.

    docs/online-softmax.md SS4 ("how running maxima change across tiles")
    is this loop, described in words, with a worked numeric example that
    is this function's own output at that example's inputs (not a
    hand-derived approximation of it). `tile_size` stands in for "how many
    key positions one streamed K/V tile covers" in the eventual CUDA
    kernels; this function's whole point is to prove the combine rule is
    correct for ANY tiling of the same row -- including tile_size=1 (a
    running update after every single element, the maximum possible number
    of max-changing events) and tile sizes that do not evenly divide the
    row length (an awkward final tile) -- before any kernel has to get it
    right under the added complexity of a CUDA thread/block launch.

    Entries equal to -inf (a causally-masked position) are supported and
    contribute exactly nothing, matching how -inf is used elsewhere in this
    repo (docs/attention-math.md SS3): a tile that is entirely -inf has
    local (max, sum) = (-inf, 0), the empty identity, and
    `combine_online_stats` leaves the running (m, l) unchanged when
    combined with it.
    """
    if scores.dim() != 1:
        raise ValueError(f"online_softmax_stats: scores must be 1-D, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("online_softmax_stats: scores must be non-empty")
    if tile_size <= 0:
        raise ValueError(f"online_softmax_stats: tile_size must be positive, got {tile_size}")

    values = scores.tolist()
    n = len(values)

    m = _NEG_INF
    l = 0.0
    for start in range(0, n, tile_size):
        tile = values[start : start + tile_size]
        tile_max = max(tile)
        if tile_max == _NEG_INF:
            tile_max, tile_sum = _NEG_INF, 0.0  # entirely-masked tile: the empty identity
        else:
            tile_sum = sum(math.exp(v - tile_max) for v in tile)
        m, l = combine_online_stats(m, l, tile_max, tile_sum)
    return m, l


def online_softmax_row(scores: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Returns the same `[S]` normalized probability vector
    `torch.softmax(scores, -1)` would produce, computed via
    `online_softmax_stats`'s running `(m, l)` instead of a direct call to
    `torch.softmax`. This isolates "does the running (m, l) this algorithm
    produces reproduce ordinary softmax's normalization constants
    correctly" from also accumulating a weighted output over V, which
    `online_softmax_attention_row` (below) tests separately.
    """
    m, l = online_softmax_stats(scores, tile_size)
    if l == 0.0:
        raise ValueError(
            "online_softmax_row: l == 0 (every entry -inf); this row is fully masked and undefined -- "
            "not a supported input (every causal row in this repo has at least one valid entry, see "
            "docs/attention-math.md SS3)"
        )
    return torch.exp(scores.to(torch.float64) - m) / l


def online_softmax_attention_row(scores: torch.Tensor, v_rows: torch.Tensor, tile_size: int) -> torch.Tensor:
    """The full V3 "unit": `softmax(scores) @ v_rows`, computed tile-by-tile
    with a running `(m, l, acc)` triple, where `acc` is an unnormalized
    running weighted sum of `v_rows` (same shape as one output row, `[D]`).

    Exactly mirroring `combine_online_stats`'s scalar identity, applied
    once to `l` (as a plain scalar) and once more, with the identical two
    correction factors, to `acc` (a vector): whenever this tile's local max
    changes the running max, BOTH the old accumulated `(l, acc)` state AND
    this tile's own newly-computed contribution get rescaled by the factor
    that makes them consistent with the new combined max, before being
    summed:

        tile_max          = max over this tile's scores
        weights            = exp(tile_scores - tile_max)             # this tile's own local softmax weights
        tile_sum           = sum(weights)
        tile_weighted_v    = weights @ tile_v                        # this tile's local (unnormalized) output

        m_new, l_new       = combine_online_stats(m, l, tile_max, tile_sum)
        old_correction      = exp(m - m_new)          if m has been set, else 0
        new_tile_correction = exp(tile_max - m_new)
        acc = acc * old_correction + tile_weighted_v * new_tile_correction

    (docs/online-softmax.md SS5 derives this exact update and proves the
    invariant it maintains: after every tile, `acc == sum(exp(scores_seen -
    m) * v_seen)` and `l == sum(exp(scores_seen - m))`, both expressed
    relative to the CURRENT running max `m`, for every position "seen" so
    far -- so at the very end, `m` is the row's true max and `acc / l` is
    exactly `softmax(scores) @ v_rows`. SS6 is that final division.)

    scores: `[S]`, one query row's raw pre-softmax logits (already scaled
        by `1/sqrt(D)` and already causal-masked with `-inf`, matching
        `flashlite.reference.attention.reference_attention`'s convention --
        this function does not apply either step itself).
    v_rows: `[S, D]`; `v_rows[j]` is the value vector for the same key
        position `scores[j]` is the score for.
    tile_size: any positive value, including 1 (every element its own
        tile -- the maximum possible number of running-max-update events)
        and values that do not evenly divide `S` (an awkward last tile).
    """
    if scores.dim() != 1:
        raise ValueError(f"online_softmax_attention_row: scores must be 1-D, got shape {tuple(scores.shape)}")
    if v_rows.dim() != 2 or v_rows.shape[0] != scores.shape[0]:
        raise ValueError(
            "online_softmax_attention_row: v_rows must be [S, D] with S == scores.shape[0]; "
            f"got scores={tuple(scores.shape)}, v_rows={tuple(v_rows.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError("online_softmax_attention_row: scores/v_rows must be non-empty")
    if tile_size <= 0:
        raise ValueError(f"online_softmax_attention_row: tile_size must be positive, got {tile_size}")

    scores64 = scores.to(torch.float64)
    v64 = v_rows.to(torch.float64)
    n, d = v64.shape

    m = _NEG_INF
    l = 0.0
    acc = torch.zeros(d, dtype=torch.float64)

    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        tile_scores = scores64[start:end]
        tile_v = v64[start:end]

        finite_mask = torch.isfinite(tile_scores)
        if not bool(finite_mask.any()):
            continue  # entirely causally-masked tile: the empty identity, (m, l, acc) unchanged

        tile_max = float(tile_scores.max())
        weights = torch.exp(tile_scores - tile_max)  # -inf entries -> exp(-inf) == 0.0, exactly
        tile_sum = float(weights.sum())
        tile_weighted_v = weights @ tile_v  # [D], relative to tile_max

        m_new, l_new = combine_online_stats(m, l, tile_max, tile_sum)
        old_correction = 0.0 if m == _NEG_INF else math.exp(m - m_new)
        new_tile_correction = math.exp(tile_max - m_new)  # tile_max <= m_new always, so this is <= 1

        acc = acc * old_correction + tile_weighted_v * new_tile_correction
        m, l = m_new, l_new

    if l == 0.0:
        raise ValueError(
            "online_softmax_attention_row: l == 0 (every entry -inf); this row is fully masked and "
            "undefined -- not a supported input"
        )
    return (acc / l).to(v_rows.dtype)
