// SAXPY: y_out[i] = a * x[i] + y_in[i]. Classic BLAS Level-1 op; paired
// with vector_add as the second Phase 0/1 "vector" kernel family (spec
// FR1: "vector/SAXPY baseline"). Same two-layer interface as vector_add
// (see vector_add.cuh for the rationale).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void saxpy_launch(float a, const float* d_x, const float* d_y_in, float* d_y_out,
                   std::size_t n, int block_size, cudaStream_t stream, dim3& out_grid,
                   dim3& out_block);

float saxpy_run_host(float a, const float* h_x, const float* h_y_in, float* h_y_out,
                      std::size_t n, int block_size = 256);

} // namespace kernelforge::kernels
