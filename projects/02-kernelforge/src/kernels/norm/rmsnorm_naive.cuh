// RMSNorm V1 (naive, one block per row, shared-memory tree reduction).
//
// Row-wise RMSNorm of an (rows x cols) row-major matrix: out[r][c] =
// in[r][c] / sqrt(mean_c(in[r][c]^2) + eps) * gamma[c]. Unlike LayerNorm,
// there is no mean-subtraction/re-centering step and no additive bias --
// see `kernelforge::reference::rmsnorm_rows`'s doc comment (this ladder's
// V0) for why. `gamma` is the per-column learned scale every real RMSNorm
// layer applies; pass an all-ones vector for pure normalization.
//
// HYPOTHESIS (written before this kernel existed):
// One thread BLOCK handles one entire row (grid.x = rows); each thread
// strides across the row (`c = tid; c < cols; c += blockDim.x`) so any
// `cols` is supported regardless of block size. The row's mean-of-squares
// is computed by a full shared-memory tree reduction (sequential
// addressing, `__syncthreads()` between every halving step -- the "simple
// CUDA first" baseline this family's spec entry calls for). Unlike
// softmax, RMSNorm needs only ONE reduction (sum of squares, not a
// separate max pass), so this kernel still makes exactly TWO full passes
// over the row's global memory: one to accumulate the sum of squares, one
// (now that the RMS is known) to normalize, scale by gamma, and write. `+
// eps` under the sqrt is this rung's numerical-stability step (guards
// against a divide-by-zero-ish blowup when a row's mean-of-squares is
// exactly or near 0 -- e.g. an all-zero row); it is held constant across
// V1-V3, the same rationale as softmax's max-subtraction (see
// src/kernels/softmax/softmax_naive.cuh's doc comment). Expect this to be
// the ladder's slowest rung, dominated by `__syncthreads()`-heavy
// shared-memory reduction rounds; V2 changes the reduction mechanism, V3
// changes how each pass loads memory. See src/kernels/norm/README.md for
// the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void rmsnorm_naive_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                           int cols, float eps, int block_size, cudaStream_t stream,
                           dim3& out_grid, dim3& out_block);

// Host convenience wrapper: allocates device buffers, copies h_in/h_gamma
// in, launches, synchronizes, copies h_out out, frees. Returns kernel-only
// elapsed ms (FR4). rows == 0 or cols == 0 are both legal and short-
// circuit without touching the GPU.
float rmsnorm_naive_run_host(const float* h_in, const float* h_gamma, float* h_out, int rows,
                              int cols, float eps, int block_size = 256);

} // namespace kernelforge::kernels
