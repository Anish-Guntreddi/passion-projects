#include "kernels/softmax/softmax_naive.cuh"

#include <cfloat>

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void softmax_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      int cols) {
  extern __shared__ float sdata[];
  const int row = static_cast<int>(blockIdx.x);
  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;
  const unsigned tid = threadIdx.x;

  // --- Pass 1: row max (shared-memory tree reduction) ---
  float local_max = -FLT_MAX;
  for (int c = static_cast<int>(tid); c < cols; c += static_cast<int>(blockDim.x)) {
    local_max = fmaxf(local_max, row_in[c]);
  }
  sdata[tid] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }
  const float row_max = sdata[0];
  __syncthreads(); // before sdata is reused for the sum reduction below

  // --- Pass 2: sum of exp(x - row_max) (second full read of the row) ---
  float local_sum = 0.0f;
  for (int c = static_cast<int>(tid); c < cols; c += static_cast<int>(blockDim.x)) {
    local_sum += expf(row_in[c] - row_max);
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  const float row_sum = sdata[0];
  __syncthreads();

  // --- Pass 3: normalize and write (third full read of the row) ---
  for (int c = static_cast<int>(tid); c < cols; c += static_cast<int>(blockDim.x)) {
    row_out[c] = expf(row_in[c] - row_max) / row_sum;
  }
}

} // namespace

void softmax_naive_launch(const float* d_in, float* d_out, int rows, int cols, int block_size,
                           cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "softmax_naive_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  softmax_naive_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, cols);
  KF_CUDA_CHECK_LAST_ERROR();
}

float softmax_naive_run_host(const float* h_in, float* h_out, int rows, int cols, int block_size) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows == 0 || cols == 0) return 0.0f; // FR6 edge size: legal, nothing to compute.
  const std::size_t bytes = n * sizeof(float);

  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  softmax_naive_launch(d_in, d_out, rows, cols, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
