#include "kernels/softmax/softmax_warp_shuffle.cuh"

#include <cfloat>

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Block-wide max reduction: every thread passes its own local partial
// `val`; every thread gets back the SAME final block-wide max. Two-level
// warp-shuffle reduction (ADR 0004: full mask, `_sync` family) -- see this
// file's .cuh header for the mechanism this replaces (V1's shared-memory
// tree). `warp_partials` is statically sized for the architectural max of
// 32 warps/block (1024 threads / 32), so it is correctly sized for any
// legal block_size without needing a dynamic second shared-memory buffer.
__device__ float block_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
  }
  __shared__ float warp_partials[32];
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_partials[warp_id] = val;
  __syncthreads();

  const unsigned num_warps = (blockDim.x + 31) / 32;
  val = (threadIdx.x < num_warps) ? warp_partials[threadIdx.x] : -FLT_MAX;
  if (warp_id == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
  }
  __shared__ float result;
  if (threadIdx.x == 0) result = val;
  __syncthreads();
  return result;
}

// Block-wide sum reduction: same two-level shape as block_reduce_max
// above, combine op is `+` and the empty-lane identity is 0.0f.
__device__ float block_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  __shared__ float warp_partials[32];
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_partials[warp_id] = val;
  __syncthreads();

  const unsigned num_warps = (blockDim.x + 31) / 32;
  val = (threadIdx.x < num_warps) ? warp_partials[threadIdx.x] : 0.0f;
  if (warp_id == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffffu, val, offset);
    }
  }
  __shared__ float result;
  if (threadIdx.x == 0) result = val;
  __syncthreads();
  return result;
}

__global__ void softmax_warp_shuffle_kernel(const float* __restrict__ in, float* __restrict__ out,
                                             int cols) {
  const int row = static_cast<int>(blockIdx.x);
  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  // --- Pass 1: row max ---
  float local_max = -FLT_MAX;
  for (int c = tid; c < cols; c += stride) local_max = fmaxf(local_max, row_in[c]);
  const float row_max = block_reduce_max(local_max);

  // --- Pass 2: sum of exp(x - row_max) ---
  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += stride) local_sum += expf(row_in[c] - row_max);
  const float row_sum = block_reduce_sum(local_sum);

  // --- Pass 3: normalize and write ---
  for (int c = tid; c < cols; c += stride) row_out[c] = expf(row_in[c] - row_max) / row_sum;
}

} // namespace

void softmax_warp_shuffle_launch(const float* d_in, float* d_out, int rows, int cols,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "softmax_warp_shuffle_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  softmax_warp_shuffle_kernel<<<grid, block, 0, stream>>>(d_in, d_out, cols);
  KF_CUDA_CHECK_LAST_ERROR();
}

float softmax_warp_shuffle_run_host(const float* h_in, float* h_out, int rows, int cols,
                                     int block_size) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows == 0 || cols == 0) return 0.0f;
  const std::size_t bytes = n * sizeof(float);

  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  softmax_warp_shuffle_launch(d_in, d_out, rows, cols, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
