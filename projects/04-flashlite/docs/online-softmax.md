# Online Softmax (Phase 4, V3)

Spec hard requirement SS1.4: this document derives the online-softmax
running update from first principles -- "this may not be hidden behind a
copied implementation" -- and every claim below is checked against
`src/flashlite/online_softmax.py` (the standalone V3 "unit," a plain
Python/PyTorch transcription with no CUDA dependency) and
`tests/math/test_online_softmax.py` (which runs entirely on CPU, no GPU
required). Written and verified **before** `src/flashlite/cuda_online_softmax/`'s
kernel exists, per the spec's literal instruction ("the V3 online-softmax
unit is implemented and tested independently before integration into
attention").

`docs/attention-math.md` SS2 already derives the *ordinary* two-pass
numerically-stable softmax (find the row max, then normalize) that V0/V1/V2
use. This document assumes that background and derives the *different*
problem online softmax solves: computing the same normalized result, and
the same softmax-weighted output, while only ever seeing the row in
sequential pieces ("tiles") -- never a complete row up front.

## 1. The problem two-pass softmax doesn't solve

Two-pass softmax needs the row's true maximum *before* it can safely start
exponentiating anything (`docs/attention-math.md` SS2). That is fine when
the whole row already sits in memory (V0/V1/V2's `scores` buffer). It stops
being fine the moment nothing materializes the whole row at all -- which is
exactly Phase 5's (V4's) design: each key/value tile arrives, gets
processed, and is gone. A softmax algorithm for that setting must be able
to answer "what is `softmax(x) @ V`?" having seen only *some* of `x` so
far, and be able to correct itself the instant it sees something bigger
than everything before it. That correction rule is this document's whole
subject.

## 2. What has to be tracked, and why

For one query row's raw scores `x_1 .. x_S` (already scaled by
`1/sqrt(D)` and already causal-masked to `-inf` where disallowed, exactly
`reference_attention`'s convention), the two numbers ordinary stable
softmax needs are:

```
m = max_k x_k                      (the row maximum)
l = sum_k exp(x_k - m)             (the normalization constant, in the m-shifted frame)
softmax(x)_j = exp(x_j - m) / l
```

and for the attention output specifically, a third quantity:

```
acc = sum_k exp(x_k - m) * V_k     (the row's un-normalized weighted output, [D])
O   = acc / l                       (= softmax(x) @ V)
```

Online softmax's job: maintain a running `(m, l, acc)` triple that is
updated incrementally as each new tile of `x` (and its matching `V` rows)
arrives, such that after the last tile, `(m, l, acc)` equal exactly the
values above -- computed from the *whole* row -- without the algorithm
ever having required the whole row to be visible at any single point in
time.

## 3. The combine identity (the one thing the whole algorithm rests on)

Suppose a row has been split into two disjoint subsets `A` and `B` (in
this project's case: two tiles, or two threads' strided element subsets,
or two half-block reduction operands -- the identity below does not care
which), each already reduced to its own local statistics `(m_A, l_A)` and
`(m_B, l_B)`, computed exactly as in SS2 but only over its own subset.
Their union's statistics are:

```
m = max(m_A, m_B)
l = l_A * exp(m_A - m) + l_B * exp(m_B - m)
```

**Why this is exact, not an approximation:** `l_A = sum_{k in A} exp(x_k -
m_A)` by construction. Multiplying it by `exp(m_A - m)` gives `sum_{k in
A} exp(x_k - m_A) * exp(m_A - m) = sum_{k in A} exp(x_k - m)` -- every term
in `A`, re-expressed relative to the *new* combined max `m`, using nothing
but algebra (`exp(a)*exp(b) = exp(a+b)`). The identical argument applies to
`B`. Summing the two re-expressed totals gives exactly `sum_{k in A union
B} exp(x_k - m)`, which is `l`'s definition for the union. **One scalar
multiply corrects an entire subset's accumulated total at once, regardless
of how many elements that subset has** -- this is the fact that makes an
*incremental* algorithm possible at all; without it, correcting for a new
maximum would require revisiting every individual element seen so far.

This identity is implemented verbatim as `combine_online_stats` in
`src/flashlite/online_softmax.py`, and its associativity/commutativity
(combining tiles in any order or grouping gives the same final `(m, l)`) is
checked directly by
`tests/math/test_online_softmax.py::test_combine_online_stats_is_associative_and_commutative`.

The same correction factor applies to a *vector* quantity, not just the
scalar `l` -- this is exactly what SS5 needs for `acc`.

## 4. Why unshifted exponentiation overflows, concretely

`exp` in float32 overflows once its argument exceeds `ln(FLT_MAX) =
ln(3.4028235e38) ~= 88.7228`. This is not a theoretical concern for this
project: `tests/correctness/test_naive_attention.py` and
`test_tiled_attention.py`'s extreme-logit cases use
`flashlite.reference.tensors.make_extreme_qkv(magnitude=80.0)`, which scales
`Q`/`K` entries into `[-80, 80)` before the `Q K^T / sqrt(D)` dot product.
Run directly (`head_dim=64`, `seed=123456789`, row 0 of a `(1, 1, 64, 64)`
extreme-QKV batch, computed on CPU in float32):

```
row min/max:        -5182.13 / 4914.68        # far past the 88.7 overflow threshold
naive (unshifted) exp(row).sum():  inf          # torch.isinf(...) is True
naive (unshifted) exp(row) / sum:  contains NaN  # inf / inf
torch.softmax(row, dim=-1):        finite, sums to 1.0  # stable softmax subtracts the max internally
```

(Reproduced exactly by evaluating `torch.exp(row)` and `torch.softmax(row,
dim=-1)` on that row; every number above is read directly off that run, not
approximated.) This is the concrete failure `docs/attention-math.md` SS2
already named ("a raw score of ~90 already overflows float32"); online
softmax has to avoid it too, while ALSO avoiding two-pass softmax's
prerequisite of knowing `m` in advance.

## 5. Why the running-max subtraction stays stable at every step

Claim: at every point during the online algorithm -- after any number of
tiles have been folded in -- every exponent ever computed is `<= 0`.

Proof by induction on tiles processed. Before any tile, there is nothing to
exponentiate. After folding in tile `t`, the running max `m^(t) =
max(m^(t-1), tile_max_t) >= tile_max_t >= x_j` for every `x_j` in tile
`t` (by definition of `tile_max_t` as that tile's own maximum) -- so every
`x_j - m^(t)` computed while processing tile `t` is `<= 0`. Separately,
`m^(t) >= m^(t-1)` (max is monotonically non-decreasing), so the
*correction* exponent `m^(t-1) - m^(t)` applied to the old accumulated
state is also `<= 0`. Both kinds of exponent used anywhere in the
algorithm are therefore always `<= 0`, hence `exp(...) in (0, 1]`,
bounded, no overflow possible -- for exactly the same shift-invariance
reason ordinary stable softmax is safe (`docs/attention-math.md` SS2), just
verified incrementally, tile by tile, instead of once for the whole row.

The one arithmetic wrinkle: the very first tile has no prior state to
correct. This project represents "no data yet" as `m^(0) = -inf, l^(0) =
0` (the identity element for SS3's combine rule). The correction factor for
that first tile is `exp(m^(0) - m^(1)) = exp(-inf - finite) = exp(-inf) =
0` -- correctly zeroes out the (nonexistent) prior contribution, using
ordinary IEEE-754 arithmetic, with one exception handled explicitly: if
BOTH operands being combined are simultaneously the empty identity (both
`-inf`), the general formula would compute `exp(-inf - -inf) = exp(nan) =
nan`; `combine_online_stats` special-cases `m == -inf` (after taking the
max) to return `(-inf, 0.0)` directly rather than letting that one
undefined case reach the exponential. This also correctly handles a fully
causally-masked tile (every entry `-inf`, local `(m, l) = (-inf, 0)`,
i.e. that tile's own empty identity): combining it with the running state
is a no-op, exactly as it should be, since a masked tile carries no
information.

## 6. How running maxima change across tiles -- a worked example

Row `x = [1.0, 5.0, 3.0, 9.0, 2.0]`, split into tiles of size 2 (`[1.0,
5.0]`, `[3.0, 9.0]`, `[2.0]` -- the true row max, `9.0`, deliberately sits
in the *middle* tile, not the first, so this example genuinely exercises a
mid-stream max update, not just a monotonically-increasing sequence).
Traced by `online_softmax_stats`'s own loop (`src/flashlite/online_softmax.py`;
every number below is that function's actual computed output for this
input, not a hand-derived approximation):

| Tile | `tile_max` | `tile_sum` (local) | running `m` before -> after | running `l` before -> after |
|---|---:|---:|---|---|
| `[1.0, 5.0]` | 5.0 | 1.0183156388887342 | `-inf -> 5.0` | `0.0 -> 1.0183156388887342` |
| `[3.0, 9.0]` | 9.0 | 1.0024787521766663 | `5.0 -> 9.0` | `1.0183156388887342 -> 1.021129853693303` |
| `[2.0]` | 2.0 | 1.0 | `9.0 -> 9.0` (unchanged) | `1.021129853693303 -> 1.0220417356588574` |

Tile 2 raises the running max from `5.0` to `9.0` -- this is where the
correction factor `exp(5.0 - 9.0) = exp(-4.0) ~= 0.0183` shrinks tile 1's
already-accumulated `l` before tile 2's own (already-`9.0`-shifted)
contribution is added. Tile 3's local max (`2.0`) is *below* the running
max, so the running max is unchanged and tile 3's own contribution gets
shrunk instead (by `exp(2.0 - 9.0) ~= 0.000912`, inside
`combine_online_stats`, not visible as a separate step in the table above
since the correction is folded into the general formula either way).

Cross-checked directly against a one-shot computation over the full row:
`max([1,5,3,9,2]) = 9.0`, `sum(exp(x - 9.0) for x in [1,5,3,9,2]) =
1.0220417356588576` -- matching the table's final `l` (`1.0220417356588574`,
identical to 15 significant digits; the last-digit difference is ordinary
float64 summation-order rounding noise, not an algorithmic error --
confirmed by re-running at `tile_size in {1, 2, 3, 5}`, i.e. every possible
tiling of this same 5-element row, all of which agree with the one-shot
value to the same precision).

This table (and every tile-size cross-check above) is exactly
`tests/math/test_online_softmax.py::test_online_softmax_stats_matches_one_shot_computation_worked_example`.

## 7. How prior partial outputs are rescaled when a new maximum appears

SS3's combine identity was stated for the scalar `l`. The identical
argument applies, term by term, to the *vector* `acc = sum exp(x_k - m) *
V_k`: multiplying a subset's already-accumulated `acc_A` by
`exp(m_A - m)` re-expresses *every* `V_k` term in that subset relative to
the new combined max `m`, for the same algebraic reason SS3 gives for `l`
(the correction factor is a scalar, so it distributes over the vector sum
term-by-term unchanged). Concretely, `online_softmax_attention_row`
(`src/flashlite/online_softmax.py`) applies, per tile:

```
tile_max          = max over this tile's scores
weights            = exp(tile_scores - tile_max)          # this tile's own local softmax weights
tile_sum           = sum(weights)
tile_weighted_v    = weights @ tile_v                       # this tile's local (unnormalized) output, [D]

m_new, l_new        = combine_online_stats(m, l, tile_max, tile_sum)
old_correction       = exp(m - m_new)            (0 if m is still -inf, i.e. nothing seen yet)
new_tile_correction  = exp(tile_max - m_new)      (<= 1, since tile_max <= m_new always)

acc = acc * old_correction + tile_weighted_v * new_tile_correction
```

Both `acc`'s existing accumulated total (`old_correction`) AND this tile's
own just-computed contribution (`new_tile_correction`) get rescaled to the
*same* new reference point `m_new` before being summed -- this is "how
prior partial outputs are rescaled when a new maximum appears" (spec
SS1.4), stated as executable code, not just described in words.

**Worked example, deliberately placing the true maximum in a middle tile,
so the rescale-of-*already-accumulated*-output case is genuinely
exercised (not just the trivial case where the max is seen first):**
`scores = [0.0, 1.0, 2.0, 3.0, 100.0, 4.0, 5.0]` (`D=8` random `V` rows,
`seed`-independent since this checks an algebraic identity, not a
statistical property), tile size 2. By tile 3 (elements `[2.0, 3.0]`), two
tiles' worth of `acc` have already accumulated relative to a running max of
`1.0`; tile 3 doesn't change the max either (local max `3.0 > 1.0`, so it
*does* trigger a rescale, of everything accumulated in tiles 1-2, by
`exp(1.0 - 3.0)`); then tile 3 (`[100.0, 4.0]`) raises the running max from
`3.0` all the way to `100.0` in one step, requiring the *entire*
already-accumulated `acc` (built from tiles 1-3, potentially many
elements) to be rescaled by one single `exp(3.0 - 100.0) ~= 1.0e-42`
multiply -- collapsing essentially all of the earlier tiles' contribution
toward zero, which is exactly correct: once a wildly larger logit appears,
softmax should assign essentially all its probability mass there, and the
algorithm accomplishes that with a single scalar correction, not by
revisiting any earlier element. Checked against `torch.softmax(scores, -1)
@ v` at every tile size from `1` (finest possible granularity) through `7`
(one single tile, no incremental behavior at all): **max absolute
difference `0.0` at every tile size** (`tests/math/test_online_softmax.py::
test_online_softmax_attention_row_matches_reference_when_max_is_in_a_later_tile`)
-- float64 arithmetic throughout this check, so this is an exact algebraic
identity being confirmed, not a within-tolerance approximation.

## 8. How the final normalized output is produced

After the last tile has been folded in, `m` is exactly the row's true
maximum and `l`/`acc` are exactly the row's true (max-shifted) `sum(exp(x_k
- m))` / `sum(exp(x_k - m) * V_k)` -- by SS3's identity applied
inductively, tile after tile, to the union of everything seen so far.
Normalizing is then a single division, done **exactly once**, not once per
tile:

```
softmax(x) @ V  =  acc / l
```

There is no other normalization step anywhere in the algorithm; every
intermediate `acc`/`l` pair during the loop is an intentionally
un-normalized, max-shifted partial total, not a partial *answer* -- dividing
early (before the last tile) would silently discard information needed to
correct for a still-possible future max increase, which is precisely the
bug this algorithm's whole design avoids.

## 9. Cross-checks performed (spec SS1.4's exit criterion: "extreme score
values and multiple tile boundaries")

All of the following are `tests/math/test_online_softmax.py`, running on
CPU only, independent of any CUDA kernel:

- **Chunking invariance**: the same row, split into every tile size from
  `1` through `len(row)` (including sizes that do not evenly divide the row
  length -- an awkward final tile, mirroring this repo's `seq_len in [1, 2,
  7, 33, 257]` convention), produces the same `(m, l)` / same output vector
  to float64 precision -- SS6's worked-example table already demonstrated
  one instance of this; the test sweeps many.
- **Extreme logits**: rows built the same way `make_extreme_qkv` builds
  them (`magnitude=80`, matching every other extreme-logit test in this
  repo), checked to remain finite and match `torch.softmax` -- while a
  parallel assertion confirms the UNSHIFTED (naive) computation on the
  identical input actually does overflow to `inf`/`nan` (SS4), so the test
  suite demonstrates the failure mode being avoided, not just the fix.
- **Max in a later tile**: SS7's worked example, generalized across
  several tile sizes and a couple of different "where is the true max"
  placements (first tile, middle tile, last tile) -- the case a
  first-tile-only bug (a plausible off-by-one: initializing `m` from the
  first tile's max instead of `-inf` before any tile) would fail on but a
  first-tile-happens-to-contain-the-max test would not catch.
- **Causal-style masking**: rows containing `-inf` entries (mirroring
  causal masking's convention, `docs/attention-math.md` SS3), including a
  tile that is entirely `-inf` (SS5's "empty identity" case), checked
  against `torch.softmax` on the same masked row.
- **Associativity/commutativity of `combine_online_stats`** (SS3):
  combining the same two partial-tile statistics in either order, or
  combining three-way in different groupings, agrees.

## 10. What Phase 4 (V3) integrates, and what it deliberately does NOT change yet

`src/flashlite/cuda_online_softmax/attention_online_softmax.cu`'s kernel 2
(`online_softmax_rows_kernel`) is this exact combine identity, realized at
two levels of the GPU's parallelism instead of a Python `for` loop: each
thread first folds its own strided subset of the row into a private
`(m, l)` via the identical per-element update (SS3's identity applied one
element at a time), then a block-level tree reduction combines every
thread's partial `(m, l)` into the row's final `(m, l)` via the identical
combine formula applied pairwise (`docs/attention-math.md`'s two-pass
`softmax_rows_kernel` for comparison: that kernel does two SEPARATE full
passes over the row, one purely for the max, one purely for the sum, because
it cannot start summing until the max is fully known; this kernel needs
only one combined max+sum pass, plus the same final normalize-and-write
pass every variant needs -- two global-memory passes over the row instead
of three).

**What V3 deliberately still does, unchanged from V2**: kernel 1
(`compute_scores_tiled_kernel`) and kernel 3 (`weighted_sum_tiled_kernel`)
are untouched, byte-for-byte V2's tiled QK^T/PV kernels -- the full `[B, H,
S, S]` score/probability matrix is **still fully materialized** in global
memory between kernels, exactly as it is in V1/V2. Phase 4's one variable
is the softmax *algorithm* (two-pass -> combined-pass, both still needing
the whole row materialized somewhere to normalize into); Phase 5 (V4,
`src/flashlite/cuda_fused/`) is what removes the `[S, S]` materialization
itself, applying this SAME per-element online recurrence directly to
`Q K^T` scores as they are computed, tile-by-tile, streamed straight into
the output accumulator with no scores buffer at all --
`docs/attention-math.md` SS5 already states this forward-reference
precisely: "V4 combines V2's tiling and V3's online softmax so the full
`S x S` matrix is never materialized at all."
