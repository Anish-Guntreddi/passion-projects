#include "kernels/norm/rmsnorm_naive.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void rmsnorm_naive_kernel(const float* __restrict__ in, const float* __restrict__ gamma,
                                      float* __restrict__ out, int cols, float eps) {
  extern __shared__ float sdata[];
  const int row = static_cast<int>(blockIdx.x);
  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;
  const unsigned tid = threadIdx.x;

  // --- Pass 1: sum of squares (shared-memory tree reduction) ---
  float local_sum_sq = 0.0f;
  for (int c = static_cast<int>(tid); c < cols; c += static_cast<int>(blockDim.x)) {
    const float v = row_in[c];
    local_sum_sq += v * v;
  }
  sdata[tid] = local_sum_sq;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  const float mean_sq = (cols > 0) ? (sdata[0] / static_cast<float>(cols)) : 0.0f;
  const float inv_rms = rsqrtf(mean_sq + eps);
  __syncthreads(); // before any thread moves on (sdata no longer needed, but keep symmetry with softmax)

  // --- Pass 2: normalize, scale by gamma, and write (second full read) ---
  for (int c = static_cast<int>(tid); c < cols; c += static_cast<int>(blockDim.x)) {
    row_out[c] = row_in[c] * inv_rms * gamma[c];
  }
}

} // namespace

void rmsnorm_naive_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                           int cols, float eps, int block_size, cudaStream_t stream,
                           dim3& out_grid, dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "rmsnorm_naive_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  rmsnorm_naive_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_gamma, d_out, cols, eps);
  KF_CUDA_CHECK_LAST_ERROR();
}

float rmsnorm_naive_run_host(const float* h_in, const float* h_gamma, float* h_out, int rows,
                              int cols, float eps, int block_size) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows == 0 || cols == 0) return 0.0f; // FR6 edge size: legal, nothing to compute.
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
  rmsnorm_naive_launch(d_in, d_gamma, d_out, rows, cols, eps, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_gamma));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
