#include "kernels/norm/rmsnorm_warp_shuffle.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Same two-level warp-shuffle block-sum shape as
// softmax_warp_shuffle.cu's block_reduce_sum (ADR 0004: full mask).
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

__global__ void rmsnorm_warp_shuffle_kernel(const float* __restrict__ in,
                                             const float* __restrict__ gamma,
                                             float* __restrict__ out, int cols, float eps) {
  const int row = static_cast<int>(blockIdx.x);
  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  // --- Pass 1: sum of squares ---
  float local_sum_sq = 0.0f;
  for (int c = tid; c < cols; c += stride) {
    const float v = row_in[c];
    local_sum_sq += v * v;
  }
  const float sum_sq = block_reduce_sum(local_sum_sq);
  const float mean_sq = (cols > 0) ? (sum_sq / static_cast<float>(cols)) : 0.0f;
  const float inv_rms = rsqrtf(mean_sq + eps);

  // --- Pass 2: normalize, scale by gamma, and write ---
  for (int c = tid; c < cols; c += stride) row_out[c] = row_in[c] * inv_rms * gamma[c];
}

} // namespace

void rmsnorm_warp_shuffle_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                                  int cols, float eps, int block_size, cudaStream_t stream,
                                  dim3& out_grid, dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "rmsnorm_warp_shuffle_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  rmsnorm_warp_shuffle_kernel<<<grid, block, 0, stream>>>(d_in, d_gamma, d_out, cols, eps);
  KF_CUDA_CHECK_LAST_ERROR();
}

float rmsnorm_warp_shuffle_run_host(const float* h_in, const float* h_gamma, float* h_out,
                                     int rows, int cols, float eps, int block_size) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows == 0 || cols == 0) return 0.0f;
  const std::size_t bytes = n * sizeof(float);
  const std::size_t gamma_bytes = static_cast<std::size_t>(cols) * sizeof(float);

  float* d_in = nullptr;
  float* d_gamma = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_gamma, gamma_bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
  KF_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, gamma_bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  rmsnorm_warp_shuffle_launch(d_in, d_gamma, d_out, rows, cols, eps, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_gamma));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
