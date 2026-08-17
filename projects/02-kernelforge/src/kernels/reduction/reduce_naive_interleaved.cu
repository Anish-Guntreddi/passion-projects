#include "kernels/reduction/reduce_naive_interleaved.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void reduce_naive_interleaved_kernel(const float* __restrict__ in,
                                                  float* __restrict__ out, std::size_t n) {
  extern __shared__ float sdata[];
  const unsigned tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  // Interleaved addressing: the active-thread predicate depends on
  // tid % (2*stride), which is NOT warp-uniform -- see header comment.
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

} // namespace

void reduce_naive_interleaved_launch(const float* d_in, float* d_out, std::size_t n,
                                      int block_size, cudaStream_t stream, dim3& out_grid,
                                      dim3& out_block, bool zero_output) {
  validate_block_size_pow2_1d(block_size, "reduce_naive_interleaved_launch");
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

  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  reduce_naive_interleaved_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

float reduce_naive_interleaved_run_host(const float* h_in, std::size_t n, float* h_sum_out,
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
  reduce_naive_interleaved_launch(d_in, d_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
