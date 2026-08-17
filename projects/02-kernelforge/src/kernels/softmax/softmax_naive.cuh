// Softmax V1 (naive, one block per row, shared-memory tree reductions).
//
// Row-wise softmax of an (rows x cols) row-major matrix: out[r][c] =
// exp(in[r][c] - max_c(in[r])) / sum_c(exp(in[r][c] - max_c(in[r]))). The
// CPU reference (`kernelforge::reference::softmax_rows`) is this ladder's
// V0.
//
// HYPOTHESIS (written before this kernel existed):
// One thread BLOCK handles one entire row (grid.x = rows); within a block,
// each thread strides across the row (`c = tid; c < cols; c += blockDim.x`)
// so any `cols` is supported regardless of block size. The row's max and
// exp-sum are each computed by a full shared-memory tree reduction
// (sequential addressing, `__syncthreads()` between every halving step --
// the "simple CUDA first" baseline this family's spec entry calls for),
// requiring THREE full passes over the row's global memory: (1) find the
// row max, (2) compute exp(x - max) and sum it, (3) recompute exp(x - max)
// once more and divide by the sum to produce the final output. The
// max-subtraction in every pass (not just the final one) is what makes
// this numerically stable -- see this file's doc comment on why it is
// held constant across every rung of this ladder, not a distinguishing
// variable. Expect this to be the ladder's slowest rung: it is bound by
// re-reading each row's data three times from global memory, on top of
// `__syncthreads()`-heavy shared-memory reductions. V2 (warp_shuffle)
// changes the reduction MECHANISM without changing the pass count; V3
// (fused_online) changes the PASS COUNT. See src/kernels/softmax/README.md
// for the measured numbers.
//
// NUMERICAL STABILITY (held constant across V1-V3, not a ladder
// variable): every rung subtracts the row max before exponentiating.
// Softmax's ladder concerns per spec FR2 are "reduction strategy, memory
// traffic, bounded fusion, numerical stability" -- the first three are
// this ladder's three distinguishing rungs; the fourth (stability) is a
// correctness PREREQUISITE present identically in all of them, the same
// way "coalesced reads" is present in every GEMM rung from V2 onward
// rather than being its own separate V2.5.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void softmax_naive_launch(const float* d_in, float* d_out, int rows, int cols, int block_size,
                           cudaStream_t stream, dim3& out_grid, dim3& out_block);

// Host convenience wrapper: allocates device buffers, copies h_in in,
// launches, synchronizes, copies h_out out, frees. Returns kernel-only
// elapsed ms (FR4). rows == 0 or cols == 0 are both legal and short-
// circuit without touching the GPU (rows == 0: nothing to allocate;
// cols == 0: every row is empty).
float softmax_naive_run_host(const float* h_in, float* h_out, int rows, int cols,
                              int block_size = 256);

} // namespace kernelforge::kernels
