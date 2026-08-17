#include "kernels/reduction/reduce_sequential_addressing.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void reduce_sequential_addressing_kernel(const float* __restrict__ in,
                                                      float* __restrict__ out, std::size_t n) {
  extern __shared__ float sdata[];
  const unsigned tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  // Sequential addressing: the ONE variable changed vs V1's interleaved
  // addressing (see header comment). tid < stride is warp-uniform for
  // every warp until stride drops below 32.
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

} // namespace

void reduce_sequential_addressing_launch(const float* d_in, float* d_out, std::size_t n,
                                          int block_size, cudaStream_t stream, dim3& out_grid,
                                          dim3& out_block, bool zero_output) {
  validate_block_size_pow2_1d(block_size, "reduce_sequential_addressing_launch");
  if (zero_output) {
    KF_CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
  }

  const std::size_t num_blocks =
      n == 0 ? 0 : (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  const dim3 grid(static_cast<unsigned>(num_blocks));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (num_blocks == 0) return;

  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  reduce_sequential_addressing_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

float reduce_sequential_addressing_run_host(const float* h_in, std::size_t n, float* h_sum_out,
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
  reduce_sequential_addressing_launch(d_in, d_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
