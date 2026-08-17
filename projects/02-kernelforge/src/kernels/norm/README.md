# RMSNorm Ladder (Phase 5)

Row-wise RMSNorm of an (rows x cols) row-major FP32 matrix:
`out[r][c] = in[r][c] / sqrt(mean_c(in[r][c]^2) + eps) * gamma[c]`. Three
GPU variants, each changing exactly one variable from the previous rung
(hard constraint 4), all validated against the same CPU reference
(`kernelforge::reference::rmsnorm_rows`, `src/common/reference_ops.cpp` —
double-accumulated for accuracy) before any timing is trusted (hard
constraint 1). Every rung's hypothesis below was written into that rung's
`.cuh` header **before** its `.cu` implementation, per
`docs/optimization-method.md`'s loop and hard constraint 10.

Spec FR2 chose "Softmax + RMSNorm **or** LayerNorm"; this repo implements
RMSNorm. Unlike LayerNorm, RMSNorm does not re-center by subtracting the
row mean first (no first reduction pass for the mean, no additive `beta`
bias after scaling) — it normalizes purely by root-mean-square magnitude,
which is both RMSNorm's defining property and the reason it needs only
ONE reduction (not two, like softmax's max+sum or LayerNorm's mean+
variance).

| Rung | File | What changes vs. previous rung | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| V1 `v1_naive` | `rmsnorm_naive.cuh/.cu` | — (baseline) | One block per row, shared-memory tree reduction for sum-of-squares, two full global-memory passes over each row (one reduction pass + one normalize-and-write pass — RMSNorm needs only one reduction, unlike softmax's two). "Simple CUDA first." |
| V2 `v2_warp_shuffle` | `rmsnorm_warp_shuffle.cuh/.cu` | Reduction mechanism | Same two-pass structure; the sum-of-squares reduction uses `__shfl_down_sync` (ADR 0004) instead of a shared-memory tree. |
| V3 `v3_vectorized` | `rmsnorm_vectorized.cuh/.cu` | Memory transaction width | Same two-pass, warp-shuffle-reduced structure as V2; both passes load/store via `float4` (128-bit transactions) instead of one float at a time. Requires `cols % 4 == 0` (validated, fails loudly otherwise). |

**Why V3 targets "memory traffic" via vectorization instead of fusion
(unlike softmax's V3):** RMSNorm already makes only one reduction pass —
there is no separate max-then-sum sequence to fuse the way softmax's V3
does. Wider (128-bit) transactions are this family's analogous "memory
traffic" mechanism instead, the same rung softmax's *own* family already
demonstrates is a distinct, real lever independent of fusion (matching
Phase 2 reduction's V3→V4 rung, `src/kernels/reduction/README.md`).

## Numerical stability (held constant, not a ladder variable)

Every rung adds `eps` under the square root before taking its reciprocal
(`rsqrtf(mean_sq + eps)`). Without it, an all-zero (or exactly-cancelling)
row's `mean_sq` is exactly 0.0, and `1/sqrt(0)` is `+inf` — every output
element in that row would be `inf * 0 = NaN`. `tests/test_rmsnorm.cpp`'s
`AllZeroRowDoesNotDivideByZero` pins this down explicitly: an all-zero
input row must produce an all-zero output row (not `NaN`/`inf`), checked
bit-exactly, not just "close to the reference".

## Gamma (learned per-column scale)

Every real RMSNorm layer applies a learned per-column scale after
normalizing; `gamma` (length `cols`) is that scale, applied identically by
every rung and by the CPU reference. There is deliberately no additive
`beta`/bias parameter — unlike LayerNorm, RMSNorm has none (see this
file's opening paragraph). Benchmarks use a random `gamma` in `[0.5,
1.5)`, not all-ones, so the multiply is exercised with genuinely varying
per-column values rather than being a no-op.

## Correctness

`tests/test_rmsnorm.cpp` checks all three variants (V1/V2 always; V3 only
when `cols % 4 == 0`, its documented constraint) against
`reference::rmsnorm_rows` on: rows = 0, cols = 0 (both legal, FR6), the
smallest legal V3 `cols` (4), a small non-multiple-of-4 `cols` (V1/V2
only, V3 deliberately excluded — see `VectorizedRejectsNonMultipleOf4Cols`
for the explicit rejection check), `cols` exactly one block, `cols` larger
than one block (multi-stride), the all-zero-row numerical-stability case
above, a realistic transformer-activation shape (2048 x 768), a
block-size sweep, and multiple seeds. See that file for the exact cases
and tolerances used.

## Benchmarks

`benchmarks/configs/rmsnorm.json` sweeps all three variants across
representative transformer hidden/feature-dimension widths (`cols`, from
128 through 8192 — every swept value is a multiple of 4, so V3 is included
at every point) at a fixed, representative `rows=4096`; results are in
`benchmarks/raw/rmsnorm.jsonl` (21 records, all `correctness_passed:
true`). See `benchmarks/methodology.md` §13 for the full as-run table and
interpretation. Headline: **V1→V2 shows the identical shape softmax's
V1→V2 did** (real win at small-medium `cols`, up to 1.38x at cols=128,
fading to parity by cols=2048-8192 — the same fixed-`__syncthreads()`-
count mechanism, independently confirmed on a second row-wise-reduction
kernel family). **V2→V3 (vectorized `float4` loads) shows a real, if
narrower-window, win than softmax's fusion rung did**: 14.5-16.2% faster
at cols=512/768 (relative to the V2 baseline), roughly level at
128/1024/2048/4096, and a small (1.9%) regression at the largest size
tested (cols=8192). V1→V3 overall peaks at
cols=512 (1.39x) and narrows to a small loss (0.98x) by cols=8192 — see
§13 for the full per-size table.
