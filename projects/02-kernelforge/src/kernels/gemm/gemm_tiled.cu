#include "kernels/gemm/gemm_tiled.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

__global__ void gemm_tiled_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, int m, int n, int k) {
  __shared__ float a_tile[kGemmTileDim][kGemmTileDim];
  __shared__ float b_tile[kGemmTileDim][kGemmTileDim];

  const int col = blockIdx.x * kGemmTileDim + threadIdx.x; // N index (V2's coalescing mapping)
  const int row = blockIdx.y * kGemmTileDim + threadIdx.y; // M index

  float acc = 0.0f;
  const int num_k_tiles = (k + kGemmTileDim - 1) / kGemmTileDim;
  for (int t = 0; t < num_k_tiles; ++t) {
    // Cooperative load: consecutive threadIdx.x -> consecutive global
    // address for BOTH loads below (a_col for A, col for B), preserving
    // V2's coalesced-read property for the loads themselves.
    const int a_col = t * kGemmTileDim + threadIdx.x;
    const int b_row = t * kGemmTileDim + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[static_cast<std::size_t>(row) * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
    __syncthreads();

    // Every element of this K-chunk's tile, read from shared memory, is
    // reused by kGemmTileDim different threads across this loop's
    // kGemmTileDim iterations combined -- the one variable this rung
    // changes vs V2 (see this file's .cuh header).
#pragma unroll
    for (int kk = 0; kk < kGemmTileDim; ++kk) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[static_cast<std::size_t>(row) * n + col] = acc;
  }
}

} // namespace

void gemm_tiled_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                        cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kGemmTileDim, kGemmTileDim);
  const dim3 grid(static_cast<unsigned>((n + kGemmTileDim - 1) / kGemmTileDim),
                   static_cast<unsigned>((m + kGemmTileDim - 1) / kGemmTileDim));
  out_grid = grid;
  out_block = block;
  if (m == 0 || n == 0) return;

  gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
  KF_CUDA_CHECK_LAST_ERROR();
}

kernelforge::OccupancyReport gemm_tiled_query_occupancy() {
  // a_tile + b_tile: both __shared__ (static), 0 dynamic shared bytes at launch.
  return kernelforge::compute_occupancy(gemm_tiled_kernel, kGemmTileDim * kGemmTileDim, 0);
}

float gemm_tiled_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {
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
  gemm_tiled_launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_a));
  KF_CUDA_CHECK(cudaFree(d_b));
  KF_CUDA_CHECK(cudaFree(d_c));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
