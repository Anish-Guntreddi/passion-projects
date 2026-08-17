// GEMM V3 (shared-memory tiling). ONE variable changes versus
// gemm_coalesced.cuh: A and B tiles are staged through shared memory and
// reused by every thread in the block, instead of every thread re-reading
// its own K-length row/column straight from global memory. Thread-to-
// output mapping (col -> threadIdx.x, row -> threadIdx.y, V2's coalescing
// fix) and one-thread-per-output-element are both unchanged.
//
// HYPOTHESIS (written before this kernel existed):
// V2 still issues M*N*K total global loads of A and M*N*K total global
// loads of B (every thread reads its own full K-length row of A and
// column of B independently -- a TILE x TILE block of threads reads the
// SAME TILE rows of A and TILE columns of B, TILE times each, all
// straight from global memory, with no reuse). This kernel instead
// cooperatively loads one TILE x TILE tile of A and one TILE x TILE tile
// of B into shared memory per K-chunk, synchronizes, then has every
// thread in the block accumulate its own output element's partial dot
// product from THAT shared tile before moving to the next K-chunk. Global
// memory traffic per output tile drops from `2 * TILE^2 * K` element
// reads (V2: every thread re-reads a full row/column) to `2 * TILE * K`
// (V3: each element of the tile is read from global memory exactly once,
// by whichever thread cooperatively loads it, and reused TILE times from
// shared memory by every thread that needs it) -- a TILE-fold reduction
// in the read pattern that reuses each global-memory value the most.
// Expect a substantial further speedup over V2, growing with matrix size
// (more K-chunks = more reuse opportunities), since this GPU's shared
// memory bandwidth is far higher than its global memory bandwidth. See
// src/kernels/gemm/README.md for the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Tile dimension for both the thread block (TILE x TILE threads) and the
// shared-memory sub-tiles of A/B (TILE x TILE floats each). 32 matches
// this GPU's warp size (coalescing granularity, same choice as
// kernels/transpose/transpose_tiled.cuh's kTransposeTileDim) and keeps
// 2 * 32 * 32 * 4 bytes = 8192 bytes of dynamic shared memory per block,
// well within this GPU's 49,152-byte default per-block budget
// (docs/gpu-model.md).
constexpr int kGemmTileDim = 32;

void gemm_tiled_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                        cudaStream_t stream, dim3& out_grid, dim3& out_block);

float gemm_tiled_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k);

} // namespace kernelforge::kernels
