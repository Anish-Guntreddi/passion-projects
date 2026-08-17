#include "kernels/vector/vector_add.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void vector_add_kernel(const float* __restrict__ x,
                                   const float* __restrict__ y,
                                   float* __restrict__ out, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + y[i];
  }
}

} // namespace

void vector_add_launch(const float* d_x, const float* d_y, float* d_out,
                        std::size_t n, int block_size, cudaStream_t stream,
                        dim3& out_grid, dim3& out_block) {
  validate_block_size_1d(block_size, "vector_add_launch");
  const dim3 block(static_cast<unsigned>(block_size));
  const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));
  out_grid = grid;
  out_block = block;
  vector_add_kernel<<<grid, block, 0, stream>>>(d_x, d_y, d_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

float vector_add_run_host(const float* h_x, const float* h_y, float* h_out,
                           std::size_t n, int block_size) {
  if (n == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.

  const std::size_t bytes = n * sizeof(float);

  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_y, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));

  KF_CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  KF_CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  vector_add_launch(d_x, d_y, d_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  KF_CUDA_CHECK(cudaFree(d_x));
  KF_CUDA_CHECK(cudaFree(d_y));
  KF_CUDA_CHECK(cudaFree(d_out));

  return elapsed_ms;
}

} // namespace kernelforge::kernels
