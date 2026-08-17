#include "kernels/gemm/gemm_coalesced.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

constexpr int kGemmCoalescedBlockDim = 32;

__global__ void gemm_coalesced_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       float* __restrict__ c, int m, int n, int k) {
  // ONE variable changed vs gemm_naive_kernel: col now maps to
  // threadIdx.x (warp-varying), row to threadIdx.y -- see this file's
  // .cuh header.
  const int col = blockIdx.x * blockDim.x + threadIdx.x; // N index -- now warp-varying
  const int row = blockIdx.y * blockDim.y + threadIdx.y; // M index -- now warp-uniform
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += a[static_cast<std::size_t>(row) * k + kk] * b[static_cast<std::size_t>(kk) * n + col];
  }
  c[static_cast<std::size_t>(row) * n + col] = acc;
}

} // namespace

void gemm_coalesced_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                            cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kGemmCoalescedBlockDim, kGemmCoalescedBlockDim);
  // Grid dims swapped vs gemm_naive_launch to match: grid.x now tiles N
  // (threadIdx.x's axis), grid.y tiles M.
  const dim3 grid(static_cast<unsigned>((n + kGemmCoalescedBlockDim - 1) / kGemmCoalescedBlockDim),
                   static_cast<unsigned>((m + kGemmCoalescedBlockDim - 1) / kGemmCoalescedBlockDim));
  out_grid = grid;
  out_block = block;
  if (m == 0 || n == 0) return;

  gemm_coalesced_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
  KF_CUDA_CHECK_LAST_ERROR();
}

float gemm_coalesced_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {
  const std::size_t c_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  if (c_elems == 0) return 0.0f;

  const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(float);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t c_bytes = c_elems * sizeof(float);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_a, a_bytes > 0 ? a_bytes : sizeof(float)));
  KF_CUDA_CHECK(cudaMalloc(&d_b, b_bytes > 0 ? b_bytes : sizeof(float)));
  KF_CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  if (a_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
  if (b_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  gemm_coalesced_launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_a));
  KF_CUDA_CHECK(cudaFree(d_b));
  KF_CUDA_CHECK(cudaFree(d_c));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
