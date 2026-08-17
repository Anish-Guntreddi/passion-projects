// Dedicated coalescing/stride microbenchmark (Phase 1 deliverable:
// "dedicated coalescing/stride microbenchmarks").
//
// Isolates the *single* variable "distance in memory between consecutive
// threads' accesses" from every other confound a full kernel (like
// transpose) has (tile size, shared memory, two array shapes, ...). Thread
// `tid` reads and writes exactly one element at flat index `tid * stride`.
// At stride=1 a warp's 32 threads touch 32 contiguous floats (one 128B
// transaction); as `stride` grows, the same 32 threads spread across up to
// 32 separate transactions while still moving the exact same number of
// *useful* bytes (one float in, one float out, per thread) -- so
// effective_bandwidth_gb_s (useful bytes / time) is a direct, clean
// measurement of the coalescing cliff, independent of any other kernel
// logic.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// `d_in`/`d_out` must have space for at least
// (num_threads - 1) * stride + 1 floats (stride_copy_buffer_elems() below
// computes the exact requirement).
void stride_copy_launch(const float* d_in, float* d_out, std::size_t num_threads,
                         int stride, int block_size, cudaStream_t stream, dim3& out_grid,
                         dim3& out_block);

float stride_copy_run_host(const float* h_in, float* h_out, std::size_t num_threads,
                            int stride, int block_size = 256);

// Minimum element count a buffer needs to safely hold every index touched
// by `stride_copy_launch(..., num_threads, stride, ...)`.
inline std::size_t stride_copy_buffer_elems(std::size_t num_threads, int stride) {
  if (num_threads == 0) return 0;
  return (num_threads - 1) * static_cast<std::size_t>(stride) + 1;
}

} // namespace kernelforge::kernels
