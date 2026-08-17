// GEMM ceiling/reference: cuBLAS SGEMM. NOT a ladder rung (spec Part 1.3,
// FR2: "cuBLAS is a ceiling/reference, never a target to beat") -- it is
// benchmarked alongside V1-V4 for CONTEXT ("how close does this repo's
// hand-written best rung get to a vendor-tuned library"), never included
// in this ladder's rung-over-rung speedup claims. See
// docs/decisions/0013-cublas-ceiling-methodology.md for the row-major/
// column-major invocation decision and why this is measured but excluded
// from the ladder narrative.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void gemm_cublas_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                         cudaStream_t stream, dim3& out_grid, dim3& out_block);

float gemm_cublas_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k);

} // namespace kernelforge::kernels
