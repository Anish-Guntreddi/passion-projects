# GEMM Ladder (Phase 4)

`C (M x N) = A (M x K) * B (K x N)`, all row-major FP32. Four GPU rungs
plus a cuBLAS ceiling/reference, each rung (V1-V4) changing exactly one
variable from the previous one (hard constraint 4), all validated against
the same CPU reference (`kernelforge::reference::gemm`,
`src/common/reference_ops.cpp` — double-accumulated for accuracy) before
any timing is trusted (hard constraint 1). Every rung's hypothesis below
was written into that rung's `.cuh` header **before** its `.cu`
implementation, per `docs/optimization-method.md`'s loop and hard
constraint 10.

| Rung | File | What changes vs. previous rung | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| V1 `v1_naive` | `gemm_naive.cuh/.cu` | — (baseline) | One thread per output element, no shared memory. Thread-to-output mapping is deliberately "the wrong way round": `row` (M) maps to the warp-varying `threadIdx.x`, leaving A reads and C writes badly uncoalesced (FR2's literal "V1 one thread/output" rung). |
| V2 `v2_coalesced` | `gemm_coalesced.cuh/.cu` | Which axis maps to `threadIdx.x` | Swaps to `col` (N) mapping to `threadIdx.x` — A's read becomes warp-uniform (broadcast), B's read and C's write become fully coalesced. No shared memory yet; pure coalescing fix. |
| V3 `v3_tiled` | `gemm_tiled.cuh/.cu` | Shared-memory tiling | 32x32 tiles of A/B staged through shared memory and reused by every thread in the block, cutting global reads from `2*TILE^2*K` to `2*TILE*K` per output tile. |
| V4 `v4_register_tiled` | `gemm_register_tiled.cuh/.cu` | Register tiling / thread coarsening | Each thread computes 8 output rows sharing one column, reusing each shared-memory B read 8x instead of 8 different threads each re-reading it once. |
| — `cublas_sgemm_ceiling` | `gemm_cublas.cuh/.cu` | N/A — **not a ladder rung** | cuBLAS SGEMM, measured for context only (spec FR2: "cuBLAS is a ceiling/reference, never a target to beat"). See `docs/decisions/0013-cublas-ceiling-methodology.md`. |

**Held constant across every GPU rung:** row-major storage and the
row-per-thread-block-tile grid convention (`grid.x` tiles N, `grid.y`
tiles M, following V2's coalescing mapping — V3/V4 preserve it exactly);
FP32 throughout (ADR 0003); no transpose of A or B (both are consumed as
given, matching the CPU reference's contract).

**Register tiling asymmetry (disclosed, not oversight):** V4 amortizes
each shared-memory B read across `kGemmRegTM=8` outputs, but NOT A's
reads — each of the 8 output rows genuinely needs a different A value.
Full 2D (`TM x TN`) register tiling, which would extend the same reuse
idea to A, is explicit future work (spec's optional "V5
vectorized/double-buffered" rung), not silently included here.

## cuBLAS ceiling

Measured alongside V1-V4 in `benchmarks/raw/gemm.jsonl` at the same sizes,
via the standard row-major/column-major transpose-and-swap invocation
(`docs/decisions/0013-cublas-ceiling-methodology.md` has the full
derivation and the one edge case it required — `K == 0` needs a direct
`cudaMemset` instead of a `cublasSgemm` call, since cuBLAS rejects
`ldb == 0` outright even though the mathematically correct K=0 result,
`C == 0`, is well-defined and every other rung's K-loop already produces
it naturally). Its number is reported as **"V4 reaches N% of cuBLAS's
throughput"**, never as a ladder rung and never as something V4 is being
measured against as a target — see `benchmarks/methodology.md` §11 for
the as-run numbers and that framing applied.

## Correctness

`tests/test_gemm.cpp` checks all five variants (V1-V4 + cuBLAS) against
`reference::gemm` on: M/N/K = 0 (each independently — FR6's "0/1 where
legal" edge case; K=0 is a legal empty-sum, M=0/N=0 are legal empty
outputs), 1x1x1, small non-tile-multiple square and rectangular shapes
(17, 32, 64, 130 — 32 is V1-V3's tile width, 64 is V4's block tile width,
so both boundaries are exercised), three-way-distinct non-tile-multiple
rectangular shapes, and a larger square shape. See that file for the
exact cases and the GEMM-specific tolerance (looser than this repo's
default — `apps/bench_gemm_main.cpp`'s `kGemmAtol`/`kGemmRtol` doc comment
explains why: legitimate fp32 accumulation-order differences across the
ladder's differently-shaped K-loops, not a correctness bug).

## Benchmarks

`benchmarks/configs/gemm.json` sweeps all five variants (V1-V4 + cuBLAS)
across a square M=N=K sweep (128 through 2048); results are in
`benchmarks/raw/gemm.jsonl` (25 records, all `correctness_passed: true`).
See `benchmarks/methodology.md` §11 for the full as-run table and
interpretation. Headline: at 2048x2048x2048, the ladder is monotonically
faster rung over rung — **V1→V2 (coalescing) is 8.26x, V2→V3 (tiling) a
further 1.22x, V3→V4 (register tiling) a further 2.66x, for a 26.8x
V1→V4 speedup overall (617.8 → 16,571.1 GFLOP/s)** — and V4 reaches
**30.8% of cuBLAS's measured throughput** at that size (reported as
exactly that framing, per `docs/decisions/
0013-cublas-ceiling-methodology.md`). V3→V4's speedup is **not**
monotonic across sizes: it is a genuine 40-44% *regression* at the two
smallest sizes tested (128, 256 — too few blocks in V4's 64x64-tile grid
to occupy this GPU's 128 SMs), crossing over to a decisive win from
1024 upward — confirming, not falsifying, a caveat the hypothesis in
`gemm_register_tiled.cuh` stated explicitly before this data existed. See
§11 for the full per-size breakdown.
