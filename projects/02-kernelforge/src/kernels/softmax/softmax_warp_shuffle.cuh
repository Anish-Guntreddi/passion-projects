// Softmax V2 (warp-shuffle block reductions). ONE variable changes versus
// softmax_naive.cuh: the row max and exp-sum are each computed with a
// warp-shuffle-based block reduction instead of a shared-memory tree with
// `__syncthreads()` between every halving step. Pass count (three full
// reads of the row), grid/block geometry, and the max-subtraction
// numerical-stability step are all otherwise identical to V1.
//
// HYPOTHESIS (written before this kernel existed):
// V1's two block-wide reductions (max, then sum) each need
// log2(block_size) rounds of shared-memory read + `__syncthreads()` +
// shared-memory write. This kernel instead reduces within each warp via
// `__shfl_down_sync` (ADR 0004: full mask, `_sync` family) -- 5 rounds of
// register-to-register shuffles, no shared memory and no barrier at all
// -- then combines the (at most 32) per-warp partial results with one more
// small warp-shuffle reduction, needing only 2 `__syncthreads()` total
// (one to publish each warp's partial to shared memory, one to publish
// the final combined value back to every thread) regardless of
// block_size, versus V1's `2 * log2(block_size)` barriers. Expect a
// modest further speedup over V1, from the same mechanism as Phase 2's
// reduce_sequential_addressing -> reduce_warp_shuffle rung (fewer
// synchronization barriers, cheaper per-element combine), NOT from any
// change in how much data is read -- both rungs still make three full
// passes over each row. See src/kernels/softmax/README.md for the
// measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void softmax_warp_shuffle_launch(const float* d_in, float* d_out, int rows, int cols,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block);

float softmax_warp_shuffle_run_host(const float* h_in, float* h_out, int rows, int cols,
                                     int block_size = 256);

} // namespace kernelforge::kernels
