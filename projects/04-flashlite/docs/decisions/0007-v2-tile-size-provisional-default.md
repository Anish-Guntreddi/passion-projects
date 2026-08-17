# ADR 0007: V2 Tile Size Provisional Default (kAttnTileDim = 32)

- **Status:** Accepted (provisional -- see "Revisit" below)
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D5 -- "Shared-memory vs.
  register-pressure tile constraints on the available GPU -> resolved
  empirically in Phase 6," but Phase 3 (tiling) needs *some* concrete tile
  size to ship a correct, benchmarked V2 kernel now. This ADR is what that
  Phase 3 default is and why, kept explicitly separate from D5's real
  answer.

## Context
`src/flashlite/cuda_tiled/attention_tiled.cu`'s two tiled kernels
(`compute_scores_tiled_kernel`, `weighted_sum_tiled_kernel`) each launch a
`kAttnTileDim x kAttnTileDim` thread block and stage `kAttnTileDim x
kAttnTileDim` shared-memory tiles of Q/K (kernel 1) or P/V (kernel 3). This
constant has to be chosen before the kernel can compile or run at all, but
the spec is explicit that the *right* value is an empirical Phase 6
question ("resolved empirically," D5), not something to guess correctly on
the first try. Phase 3's job (roadmap: "Tiled implementation correct and
benchmarked") only requires a value that is correct and reasonably
justified, not the tuned optimum.

## Decision
`kAttnTileDim = 32` (`src/flashlite/cuda_tiled/attention_tiled.cuh`), for
three reasons, none of which is "this is the fastest value" (that claim is
explicitly deferred to Phase 6):

1. **Matches this GPU's warp size.** Every coalescing argument in this
   kernel's header comment (`attention_tiled.cuh`) is stated in terms of a
   32-thread warp reading 32 consecutive elements in one transaction; a
   tile dimension equal to the warp size is what makes that argument exact
   rather than approximate.
2. **Matches KernelForge's own `gemm_tiled` precedent.**
   `../02-kernelforge/src/kernels/gemm/gemm_tiled.cuh` uses
   `kGemmTileDim = 32` for the structurally identical
   cooperative-tile-load pattern this kernel's kernel 3
   (`weighted_sum_tiled_kernel`) is a direct instance of (P @ V is a plain
   GEMM); reusing the same starting value keeps the two sibling projects'
   "first tiled attempt" directly comparable instead of introducing an
   unexplained difference.
3. **Fits comfortably inside this GPU's shared-memory budget.** Two
   `32 x 32 x 4`-byte tiles per kernel is `8,192` bytes/block, versus a
   `49,152`-byte default per-block budget (`docs/gpu-model.md` in
   `../02-kernelforge`, same GPU) -- under 17% of the budget, so shared
   memory capacity is not a constraint this choice needs to trade off
   against yet (unlike, say, `transpose_tiled`'s or a much larger attention
   tile's bank-conflict/occupancy tradeoffs, which Phase 6 is exactly where
   that kind of tradeoff gets measured).

`32 x 32` threads = 1024 threads/block, which is this GPU's
`maxThreadsPerBlock` exactly (`docs/gpu-model.md`) -- one full block, at
most one resident per SM for this launch config given its register/shared
usage. This is an accepted Phase 3 characteristic, not a hidden problem:
Phase 6's tile-size sweep will directly measure whether a smaller tile
(more blocks resident per SM, more latency-hiding) outperforms this one.

## Consequences
- V2 is correct for every tested shape regardless of whether `seq_len` or
  `head_dim` is a multiple of 32 -- both tiled kernels zero-pad out-of-bounds
  tile loads exactly the way `gemm_tiled_kernel` does
  (`row < S && d_idx < D` / `key_row_for_load < S && d_idx < D` guards),
  verified directly by `tests/correctness/test_tiled_attention.py`'s
  `seq_len in [1, 2, 7, 33, 257]` sweep (none of which is a multiple of 32).
- `benchmarks/schema/bench_result.schema.json`'s new optional `tile_size`
  field (ADR 0008) records `32` on every committed `v2_tiled` result, so a
  reader of `benchmarks/raw/attention.jsonl` never has to cross-reference
  this ADR to know what was actually run.
- **Revisit in Phase 6.** D5 remains formally unresolved until Phase 6's
  tile-size/block-shape sweep (roadmap: "Tile-size/block-shape sweep for
  supported shape family; profile throughput, occupancy, stalls" ->
  "Tuned defaults based on benchmark evidence, not folklore"). If that
  sweep finds a different value measurably better, `kAttnTileDim` changes
  and this ADR is superseded, not edited in place -- consistent with how
  `docs/decisions/` already treats every other spec-driven decision in this
  repo.
