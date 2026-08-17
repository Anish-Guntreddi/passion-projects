// RMSNorm V3 (vectorized float4 loads). ONE variable changes versus
// rmsnorm_warp_shuffle.cuh: both passes over the row load/store four
// floats at a time via `float4` (128-bit transactions) instead of one
// float at a time. Reduction mechanism (warp-shuffle, carried over
// unchanged from V2), pass count (still two full passes), and the `+ eps`
// numerical-stability step are all otherwise identical to V2.
//
// HYPOTHESIS (written before this kernel existed):
// V2 issues one 32-bit global-memory transaction per thread per element.
// This kernel instead has each thread load/store 4 consecutive elements
// per instruction via `float4` -- a single 128-bit-wide memory
// transaction for what used to be 4 separate 32-bit ones, the same
// mechanism as Phase 2's reduce_sequential_addressing/warp_shuffle ->
// reduce_vectorized_coarsened rung (see that file's hypothesis). This
// requires `cols` to be a multiple of 4 (validated, fails loudly
// otherwise -- see common/launch_validate.hpp's
// validate_cols_multiple_of_4 for the alignment reasoning). Expect a
// further speedup over V2, growing with `cols` (more transactions
// consolidated per row), from fewer, wider memory requests per SM per
// unit of data moved -- a "memory traffic" mechanism, not a reduction
// mechanism, distinguishing this rung's variable from V1->V2's. See
// src/kernels/norm/README.md for the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void rmsnorm_vectorized_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                                int cols, float eps, int block_size, cudaStream_t stream,
                                dim3& out_grid, dim3& out_block);

// Throws std::invalid_argument if cols is not a multiple of 4 (see this
// file's doc comment / common/launch_validate.hpp).
float rmsnorm_vectorized_run_host(const float* h_in, const float* h_gamma, float* h_out, int rows,
                                   int cols, float eps, int block_size = 256);

} // namespace kernelforge::kernels
