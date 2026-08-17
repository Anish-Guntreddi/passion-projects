// RMSNorm V2 (warp-shuffle block reduction). ONE variable changes versus
// rmsnorm_naive.cuh: the sum-of-squares reduction uses a warp-shuffle
// block reduction instead of a shared-memory tree with `__syncthreads()`
// between every halving step. Pass count (two full reads of the row),
// grid/block geometry, and the `+ eps` numerical-stability step are all
// otherwise identical to V1.
//
// HYPOTHESIS (written before this kernel existed):
// Same mechanism as softmax_warp_shuffle.cuh's hypothesis, applied to
// RMSNorm's single reduction: `__shfl_down_sync` (ADR 0004: full mask)
// replaces log2(block_size) shared-memory + `__syncthreads()` rounds with
// 5 in-register shuffle rounds plus one small cross-warp combine, cutting
// this kernel's barrier count from log2(block_size) down to 2 regardless
// of block_size. Expect a modest further speedup over V1, from cheaper
// synchronization, NOT from any change in memory traffic (both rungs
// still make two full passes over each row). See
// src/kernels/norm/README.md for the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void rmsnorm_warp_shuffle_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                                  int cols, float eps, int block_size, cudaStream_t stream,
                                  dim3& out_grid, dim3& out_block);

float rmsnorm_warp_shuffle_run_host(const float* h_in, const float* h_gamma, float* h_out,
                                     int rows, int cols, float eps, int block_size = 256);

} // namespace kernelforge::kernels
