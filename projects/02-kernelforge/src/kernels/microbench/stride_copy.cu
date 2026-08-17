#include "kernels/microbench/stride_copy.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void stride_copy_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    std::size_t num_threads, int stride) {
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < num_threads) {
    const std::size_t idx = tid * static_cast<std::size_t>(stride);
    out[idx] = in[idx];
  }
}

} // namespace

void stride_copy_launch(const float* d_in, float* d_out, std::size_t num_threads,
                         int stride, int block_size, cudaStream_t stream, dim3& out_grid,
                         dim3& out_block) {
  validate_block_size_1d(block_size, "stride_copy_launch");
  const dim3 block(static_cast<unsigned>(block_size));
  const dim3 grid(static_cast<unsigned>((num_threads + block_size - 1) / block_size));
  out_grid = grid;
  out_block = block;
  stride_copy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, num_threads, stride);
  KF_CUDA_CHECK_LAST_ERROR();
}

float stride_copy_run_host(const float* h_in, float* h_out, std::size_t num_threads,
                            int stride, int block_size) {
  if (num_threads == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.
  const std::size_t buffer_elems = stride_copy_buffer_elems(num_threads, stride);
  const std::size_t bytes = buffer_elems * sizeof(float);

  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
  // out may contain caller-chosen sentinel values for untouched indices;
  // copy them over so the post-copy comparison of untouched indices is
  // meaningful (proves the kernel didn't touch anything it shouldn't).
  KF_CUDA_CHECK(cudaMemcpy(d_out, h_out, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  stride_copy_launch(d_in, d_out, num_threads, stride, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
