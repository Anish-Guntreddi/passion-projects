#include "kernels/reduction/reduce_naive_global.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// No shared memory, no tree reduction: every in-bounds thread performs
// exactly one global atomicAdd of its own element. See reduce_naive_global.
// cuh's header for why this (not reduce_naive_interleaved_kernel) is the
// ladder's true "naive global-memory" baseline.
__global__ void reduce_naive_global_kernel(const float* __restrict__ in, float* __restrict__ out,
                                            std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out, in[i]);
  }
}

} // namespace

void reduce_naive_global_launch(const float* d_in, float* d_out, std::size_t n, int block_size,
                                 cudaStream_t stream, dim3& out_grid, dim3& out_block,
                                 bool zero_output) {
  // Unlike every other reduction rung, this kernel has no shared-memory
  // tree step, so it does not require a power-of-two block_size -- only the
  // generic 1-1024 bound (see this file's .cuh header).
  validate_block_size_1d(block_size, "reduce_naive_global_launch");
  if (zero_output) {
    KF_CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
  }

  const std::size_t num_blocks =
      n == 0 ? 0 : (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  const dim3 grid(static_cast<unsigned>(num_blocks));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (num_blocks == 0) return; // n == 0: d_out is already 0 from the memset above.

  reduce_naive_global_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

float reduce_naive_global_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                    int block_size) {
  if (n == 0) {
    *h_sum_out = 0.0f;
    return 0.0f;
  }

  const std::size_t bytes = n * sizeof(float);
  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  reduce_naive_global_launch(d_in, d_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
