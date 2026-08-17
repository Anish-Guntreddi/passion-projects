#include "kernels/vector/saxpy.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void saxpy_kernel(float a, const float* __restrict__ x,
                              const float* __restrict__ y_in, float* __restrict__ y_out,
                              std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y_out[i] = a * x[i] + y_in[i];
  }
}

} // namespace

void saxpy_launch(float a, const float* d_x, const float* d_y_in, float* d_y_out,
                   std::size_t n, int block_size, cudaStream_t stream, dim3& out_grid,
                   dim3& out_block) {
  validate_block_size_1d(block_size, "saxpy_launch");
  const dim3 block(static_cast<unsigned>(block_size));
  const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));
  out_grid = grid;
  out_block = block;
  saxpy_kernel<<<grid, block, 0, stream>>>(a, d_x, d_y_in, d_y_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

float saxpy_run_host(float a, const float* h_x, const float* h_y_in, float* h_y_out,
                      std::size_t n, int block_size) {
  if (n == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.

  const std::size_t bytes = n * sizeof(float);

  float* d_x = nullptr;
  float* d_y_in = nullptr;
  float* d_y_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_x, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_y_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_y_out, bytes));

  KF_CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  KF_CUDA_CHECK(cudaMemcpy(d_y_in, h_y_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  saxpy_launch(a, d_x, d_y_in, d_y_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_y_out, d_y_out, bytes, cudaMemcpyDeviceToHost));

  KF_CUDA_CHECK(cudaFree(d_x));
  KF_CUDA_CHECK(cudaFree(d_y_in));
  KF_CUDA_CHECK(cudaFree(d_y_out));

  return elapsed_ms;
}

} // namespace kernelforge::kernels
