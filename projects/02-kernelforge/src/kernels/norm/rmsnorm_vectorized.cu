#include "kernels/norm/rmsnorm_vectorized.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Copy-identical shape to rmsnorm_warp_shuffle.cu's block_reduce_sum --
// the ONLY variable this rung changes vs V2 is the loading/storing phase
// below, not the reduction mechanism (see this file's .cuh header).
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

__global__ void rmsnorm_vectorized_kernel(const float* __restrict__ in,
                                           const float* __restrict__ gamma,
                                           float* __restrict__ out, int cols, float eps) {
  const int row = static_cast<int>(blockIdx.x);
  // Safe: cols % 4 == 0 is validated before launch, so every row starts
  // 16-byte aligned (see common/launch_validate.hpp::validate_cols_multiple_of_4).
  const float4* row_in4 = reinterpret_cast<const float4*>(in + static_cast<std::size_t>(row) * cols);
  const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
  float4* row_out4 = reinterpret_cast<float4*>(out + static_cast<std::size_t>(row) * cols);
  const int cols4 = cols / 4;
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  // --- Pass 1: sum of squares, float4 loads ---
  float local_sum_sq = 0.0f;
  for (int c4 = tid; c4 < cols4; c4 += stride) {
    const float4 v = row_in4[c4];
    local_sum_sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
  }
  const float sum_sq = block_reduce_sum(local_sum_sq);
  const float mean_sq = (cols > 0) ? (sum_sq / static_cast<float>(cols)) : 0.0f;
  const float inv_rms = rsqrtf(mean_sq + eps);

  // --- Pass 2: normalize, scale by gamma, and write, float4 loads/stores ---
  for (int c4 = tid; c4 < cols4; c4 += stride) {
    const float4 v = row_in4[c4];
    const float4 g = gamma4[c4];
    float4 o;
    o.x = v.x * inv_rms * g.x;
    o.y = v.y * inv_rms * g.y;
    o.z = v.z * inv_rms * g.z;
    o.w = v.w * inv_rms * g.w;
    row_out4[c4] = o;
  }
}

} // namespace

void rmsnorm_vectorized_launch(const float* d_in, const float* d_gamma, float* d_out, int rows,
                                int cols, float eps, int block_size, cudaStream_t stream,
                                dim3& out_grid, dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "rmsnorm_vectorized_launch");
  validate_cols_multiple_of_4(cols, "rmsnorm_vectorized_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  rmsnorm_vectorized_kernel<<<grid, block, 0, stream>>>(d_in, d_gamma, d_out, cols, eps);
  KF_CUDA_CHECK_LAST_ERROR();
}

float rmsnorm_vectorized_run_host(const float* h_in, const float* h_gamma, float* h_out, int rows,
                                   int cols, float eps, int block_size) {
  validate_cols_multiple_of_4(cols, "rmsnorm_vectorized_run_host");
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
  rmsnorm_vectorized_launch(d_in, d_gamma, d_out, rows, cols, eps, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_gamma));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
