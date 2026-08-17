// Transpose V1 (naive): out[c][r] = in[r][c], one thread per element, no
// shared memory. `in` is rows x cols row-major, `out` is cols x rows
// row-major.
//
// This is the Phase 1 "pathological access" half of the coalescing
// demonstration (see transpose_tiled.cuh for the "contiguous access"
// half). Within a warp, consecutive threads have consecutive threadIdx.x,
// which is mapped to consecutive input columns:
//   - the READ (`in[row*cols+col]`) is coalesced: 32 consecutive threads
//     read 32 consecutive floats -> one 128B transaction.
//   - the WRITE (`out[col*rows+row]`) is NOT coalesced: 32 consecutive
//     threads write to addresses `rows` floats apart -> up to 32 separate
//     transactions for one warp.
// See docs/decisions and benchmarks/methodology.md ("Phase 1 results") for
// the measured before/after against transpose_tiled.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void transpose_naive_launch(const float* d_in, float* d_out, int rows, int cols,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block);

float transpose_naive_run_host(const float* h_in, float* h_out, int rows, int cols);

} // namespace kernelforge::kernels
