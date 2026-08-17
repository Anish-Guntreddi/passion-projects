# Attention Math (V0/V1 baseline)

This document derives the mathematical invariant every variant in the
ladder (V0-V4, spec Part 1 SS1.3) is checked against, and works out the
compute/memory cost of the *materialized* form implemented in Phases 0-1.
It deliberately stops short of the online-softmax derivation -- that is
`docs/online-softmax.md`'s job, written independently before the V3 kernel
is implemented (Phase 4, spec SS1.4), not duplicated here.

## 1. Definition

For one (batch, head) slice, given queries `Q`, keys `K`, values `V`, each
`[S, D]` (sequence length `S`, head dimension `D`):

```
S_raw = Q K^T / sqrt(D)              # [S, S], "raw scores"
S_masked[i, j] = S_raw[i, j]  if not causal or j <= i
               = -inf         if causal and j > i
P = softmax(S_masked, dim=-1)        # [S, S], row-wise softmax
O = P V                               # [S, D]
```

Extended to the batched, multi-head layout this project uses
(`[B, H, S, D]`, ADR 0002), every one of `B * H` (batch, head) slices is
independent -- the definition above applies per-slice, unchanged.

## 2. Why row-wise softmax, and why it needs a maximum subtracted

`softmax(x)_j = exp(x_j) / sum_k exp(x_k)`. Applied to a full row of `S`
(dim=-1) turns the row's raw scores into a probability distribution over
key positions: non-negative, summing to 1. Every test in
`tests/math/test_reference_attention.py::test_shape_and_dtype_sweep_produces_finite_normalized_output`
checks this literally (`probs.sum(dim=-1) == 1`).

`exp` grows fast: a raw score of ~90 already overflows float32
(`exp(89) ~ 4.5e38`, close to `FLT_MAX ~ 3.4e38`; the reference project's
`make_extreme_qkv` helper deliberately constructs scores in that range to
exercise this). Since softmax is shift-invariant --
`softmax(x)_j == softmax(x - c)_j` for any constant `c`, because the `c`
term cancels between numerator and denominator -- subtracting the row
maximum before exponentiating (`c = max(x)`) makes every exponent `<= 0`,
so every `exp(...)` term is in `(0, 1]` and cannot overflow, while the
result is mathematically identical. `torch.softmax` (V0) does this
internally; the naive CUDA kernel (V1, `softmax_rows_kernel` in
`attention_naive.cu`) does it explicitly and by hand, as a two-pass
block-per-row reduction (find the row max, then find the row sum of
`exp(x - max)`), because it has no built-in softmax to call.

This max-subtraction is the *ordinary* numerically-stable softmax, applied
to a fully materialized row. It is **not** the *online* softmax Phase 4
introduces -- that variant's problem is different: it needs the running
max to update correctly as new key/value tiles stream in *before the whole
row is known*, without ever materializing the full row. `docs/online-softmax.md`
derives that update rule; this file's V0/V1 only ever see a complete row at
once, so the ordinary two-pass form is sufficient and simpler.

## 3. Causal masking

`causal=True` restricts query position `i` to attend only to key positions
`j <= i` (position `i` cannot see the future). Implemented as
`S_masked[i, j] = -inf` for `j > i` before the softmax -- `exp(-inf) == 0`,
so those positions receive exactly zero probability without changing the
normalization of the valid positions. Every causal row has at least one
valid entry (`j == i`), so no row is ever entirely masked (no
divide-by-zero risk in the softmax denominator) -- verified directly by
`test_causal_row_zero_only_attends_to_itself` (row 0 can only attend to
itself, and its output must equal `V[0, :]` exactly) and
`test_causal_output_does_not_depend_on_future_keys` (perturbing a future
key/value must not change an earlier row's output).

## 4. Compute and memory cost of the *materialized* form (V0/V1)

Per (batch, head) slice:

- **`Q K^T`**: `S x S` output, each entry a `D`-term dot product ->
  `2 * S^2 * D` FLOPs (`D` multiplies + `D` adds, per entry).
- **`P V`**: `S x D` output, each entry an `S`-term dot product ->
  `2 * S^2 * D` FLOPs.
- **Total compute**: `4 * S^2 * D` FLOPs per slice (`scripts/run_benchmarks.py`
  uses this exact formula for the `gflops` field of every committed
  `BenchResult`).
- **Materialized memory**: the full `S x S` score/probability matrix is
  written to memory TWICE (written by `compute_scores_kernel`, then
  overwritten in place by `softmax_rows_kernel`) and read back once more by
  `weighted_sum_kernel` -- `O(S^2)` memory traffic per slice, on top of the
  `O(S * D)` traffic actually irreducibly required to read `Q`, `K`, `V`
  once and write `O` once.

For `S >> D` (long sequences, the regime this project's whole premise is
about), `O(S^2)` traffic dominates `O(S * D)` traffic -- this is *exactly*
the wasted memory movement the spec's overview describes ("attention
performance is dominated by memory movement, not FLOP count alone", spec
SS1.1). Phase 2 (memory accounting) quantifies this precisely, before
Phase 3 (tiling) and Phase 5 (fusion) remove it; `benchmarks/methodology.md`
SS5's `bytes_moved` field for this project deliberately counts only the
*irreducible* `Q + K + V + O` traffic, so the gap between that number's
implied bandwidth and the *measured* effective bandwidth of the naive
kernel is itself the evidence for this section's claim.

## 5. What V0 and V1 do NOT do (forward references)

- **V0** (`flashlite.reference.attention`): the direct transcription of
  section 1 above using `torch.matmul` + `torch.softmax`. No tiling, no
  fusion -- it is allowed to materialize everything, because its only job
  is to be the trusted, "obviously correct by inspection" ground truth
  (Phase 0 exit criterion).
- **V1** (`flashlite._cuda_naive`): the same math, by hand, in three CUDA
  kernels, still fully materializing `S`/`P` in global memory between
  kernels (`docs/attention-math.md` SS4's cost applies to it directly) --
  the point of V1 is "does a hand-written kernel reproduce V0's numbers",
  not efficiency (Phase 1 exit criterion).
- **V2** (tiled, Phase 3) keeps this section's math identical but computes
  it tile-by-tile through shared memory, without changing what gets
  computed.
- **V3** (online softmax, Phase 4) changes *how* the softmax normalization
  is accumulated (running max + running sum, `docs/online-softmax.md`) so
  it can be computed incrementally across K/V tiles instead of needing a
  complete row first.
- **V4** (fused, Phase 5) combines V2's tiling and V3's online softmax so
  the full `S x S` matrix is never materialized at all -- `O(S * D)` memory
  traffic per slice, matching the irreducible lower bound this section
  identified.
