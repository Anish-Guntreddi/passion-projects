#include "kernels/gemm/gemm_naive.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

// Fixed 32x32 thread block -- not exposed via the generic CLI --block-size
// (same precedent as kernels/transpose/transpose_tiled.cuh's
// kTransposeTileDim). threadIdx.x maps to `row` (the M dimension) here --
// see this file's .cuh header for why that is the deliberately naive
// choice this rung makes.
constexpr int kGemmNaiveBlockDim = 32;

__global__ void gemm_naive_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, int m, int n, int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x; // M index -- warp-varying (see header)
  const int col = blockIdx.y * blockDim.y + threadIdx.y; // N index -- warp-uniform (see header)
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += a[static_cast<std::size_t>(row) * k + kk] * b[static_cast<std::size_t>(kk) * n + col];
  }
  c[static_cast<std::size_t>(row) * n + col] = acc;
}

} // namespace

void gemm_naive_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                        cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kGemmNaiveBlockDim, kGemmNaiveBlockDim);
  const dim3 grid(static_cast<unsigned>((m + kGemmNaiveBlockDim - 1) / kGemmNaiveBlockDim),
                   static_cast<unsigned>((n + kGemmNaiveBlockDim - 1) / kGemmNaiveBlockDim));
  out_grid = grid;
  out_block = block;
  if (m == 0 || n == 0) return;

  gemm_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
  KF_CUDA_CHECK_LAST_ERROR();
}

float gemm_naive_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {
  const std::size_t c_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  if (c_elems == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.

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
  gemm_naive_launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_a));
  KF_CUDA_CHECK(cudaFree(d_b));
  KF_CUDA_CHECK(cudaFree(d_c));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
