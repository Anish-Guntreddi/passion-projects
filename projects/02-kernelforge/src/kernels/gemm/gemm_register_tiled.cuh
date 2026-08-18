// GEMM V4 (register tiling / thread coarsening). ONE variable changes
// versus gemm_tiled.cuh: each thread now computes kGemmRegTM (8) output
// elements along the M dimension instead of exactly 1, reusing every
// value it reads out of shared memory kGemmRegTM times instead of once.
// Block/grid-level shared-memory tiling (loading BM x BK / BK x BN tiles
// of A/B once per K-chunk) is otherwise the same idea as V3, just with
// larger tile dimensions (BM=BN=64, BK=16) to keep the per-thread:
// per-output-element ratio favorable at kGemmRegTM=8.
//
// HYPOTHESIS (written before this kernel existed):
// V3 gives each thread exactly one output element; for every K-chunk, that
// thread reads kGemmTileDim values out of a_tile and kGemmTileDim values
// out of b_tile, but the b_tile READ for a given (kk, col) is IDENTICAL
// for every thread sharing that `col` (kGemmTileDim of them, one per
// distinct `row` in the tile) -- V3 does not exploit that: each of those
// threads independently re-reads the same shared-memory value. This
// kernel instead has each thread own kGemmRegTM=8 output rows sharing the
// SAME `col` (and therefore the same per-thread b_tile shared-memory
// read), so it can read one b_tile value into a register ONCE per K-step
// and reuse it across all 8 of its accumulators, instead of 8 separate
// threads each independently re-reading the identical shared-memory
// address. Shared-memory read traffic per useful FMA drops by roughly a
// factor of kGemmRegTM for the B operand (A's per-thread reads do not
// shrink the same way, since each of the 8 rows genuinely needs a
// DIFFERENT A value -- this is a real, disclosed asymmetry, not
// oversight: full 2D (TM x TN) register tiling would extend the same
// idea to A's reads too and is explicit future work, not V5). Expect a
// further speedup over V3 at every size where V3 was already
// shared-memory-bandwidth-limited rather than occupancy-limited (larger
// matrices, where the K-loop dominates total kernel time), from fewer
// shared-memory instructions issued per unit of arithmetic work. See
// src/kernels/gemm/README.md for the measured numbers.
#pragma once

#include <cuda_runtime.h>

#include "common/occupancy.cuh"

namespace kernelforge::kernels {

// Block-level tile dimensions (A: BM x BK staged tile, B: BK x BN staged
// tile) and per-thread register-tile height (each thread accumulates
// kGemmRegTM output elements sharing one `col`).
//
// 2 * (kGemmRegBM * kGemmRegBK + kGemmRegBK * kGemmRegBN) * 4 bytes
// = 2 * (64*16 + 16*64) * 4 = 8192 bytes of dynamic... actually STATIC
// shared memory per block (kept well under this GPU's 49,152-byte
// default per-block budget, docs/gpu-model.md, same headroom V3 has).
constexpr int kGemmRegBM = 64;
constexpr int kGemmRegBN = 64;
constexpr int kGemmRegBK = 16;
constexpr int kGemmRegTM = 8;
// (kGemmRegBM / kGemmRegTM) distinct "thread rows" x kGemmRegBN distinct
// "thread cols" = one thread per (thread-row, thread-col) pair, laid out
// as a 1D block (not 2D dim3) so threadIdx.x alone determines both via
// division/modulo -- see gemm_register_tiled.cu.
constexpr int kGemmRegBlockThreads = (kGemmRegBM / kGemmRegTM) * kGemmRegBN;

void gemm_register_tiled_launch(const float* d_a, const float* d_b, float* d_c, int m, int n,
                                 int k, cudaStream_t stream, dim3& out_grid, dim3& out_block);

float gemm_register_tiled_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n,
                                    int k);

// Phase 6: theoretical occupancy of gemm_register_tiled_kernel at its fixed
// kGemmRegBlockThreads = 512-thread block shape -- see common/occupancy.cuh
// and profiling/case-studies/03-gemm-register-tiling-occupancy.md.
kernelforge::OccupancyReport gemm_register_tiled_query_occupancy();

} // namespace kernelforge::kernels
