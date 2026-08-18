#include "kernels/gemm/gemm_register_tiled.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

__global__ void gemm_register_tiled_kernel(const float* __restrict__ a,
                                            const float* __restrict__ b, float* __restrict__ c,
                                            int m, int n, int k) {
  __shared__ float a_tile[kGemmRegBM][kGemmRegBK];
  __shared__ float b_tile[kGemmRegBK][kGemmRegBN];

  const int tid = static_cast<int>(threadIdx.x); // 1D block, see .cuh header
  // (kGemmRegBM / kGemmRegTM) distinct thread-rows x kGemmRegBN distinct
  // thread-cols. thread_col is the FASTEST-varying (tid % BN), matching
  // V2/V3's coalescing convention (warp-varying axis = N).
  const int thread_row = tid / kGemmRegBN;
  const int thread_col = tid % kGemmRegBN;

  const int block_row0 = blockIdx.y * kGemmRegBM; // this block's M origin
  const int col = blockIdx.x * kGemmRegBN + thread_col; // this thread's N index (fixed)

  float acc[kGemmRegTM];
#pragma unroll
  for (int i = 0; i < kGemmRegTM; ++i) acc[i] = 0.0f;

  const int num_k_tiles = (k + kGemmRegBK - 1) / kGemmRegBK;
  for (int t = 0; t < num_k_tiles; ++t) {
    const int k0 = t * kGemmRegBK;

    // Cooperative load, strided across all kGemmRegBlockThreads threads
    // (a_tile has BM*BK elements, b_tile has BK*BN -- each thread loads
    // exactly BM*BK/kGemmRegBlockThreads and BK*BN/kGemmRegBlockThreads
    // of them respectively, both of which are exact integer divisions for
    // this file's fixed BM/BN/BK/TM constants).
    for (int idx = tid; idx < kGemmRegBM * kGemmRegBK; idx += kGemmRegBlockThreads) {
      const int r = idx / kGemmRegBK;
      const int c_local = idx % kGemmRegBK;
      const int g_row = block_row0 + r;
      const int g_col = k0 + c_local;
      a_tile[r][c_local] =
          (g_row < m && g_col < k) ? a[static_cast<std::size_t>(g_row) * k + g_col] : 0.0f;
    }
    for (int idx = tid; idx < kGemmRegBK * kGemmRegBN; idx += kGemmRegBlockThreads) {
      const int r = idx / kGemmRegBN;
      const int c_local = idx % kGemmRegBN;
      const int g_row = k0 + r;
      const int g_col = blockIdx.x * kGemmRegBN + c_local;
      b_tile[r][c_local] =
          (g_row < k && g_col < n) ? b[static_cast<std::size_t>(g_row) * n + g_col] : 0.0f;
    }
    __syncthreads();

    // ONE variable changed vs V3: b_val is read from shared memory ONCE
    // per (kk, thread) and reused across all kGemmRegTM accumulators,
    // instead of kGemmRegTM different threads each re-reading it
    // independently (see this file's .cuh header).
#pragma unroll
    for (int kk = 0; kk < kGemmRegBK; ++kk) {
      const float b_val = b_tile[kk][thread_col];
#pragma unroll
      for (int i = 0; i < kGemmRegTM; ++i) {
        acc[i] += a_tile[thread_row * kGemmRegTM + i][kk] * b_val;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kGemmRegTM; ++i) {
    const int row = block_row0 + thread_row * kGemmRegTM + i;
    if (row < m && col < n) {
      c[static_cast<std::size_t>(row) * n + col] = acc[i];
    }
  }
}

} // namespace

void gemm_register_tiled_launch(const float* d_a, const float* d_b, float* d_c, int m, int n,
                                 int k, cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kGemmRegBlockThreads);
  const dim3 grid(static_cast<unsigned>((n + kGemmRegBN - 1) / kGemmRegBN),
                   static_cast<unsigned>((m + kGemmRegBM - 1) / kGemmRegBM));
  out_grid = grid;
  out_block = block;
  if (m == 0 || n == 0) return;

  gemm_register_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
  KF_CUDA_CHECK_LAST_ERROR();
}

kernelforge::OccupancyReport gemm_register_tiled_query_occupancy() {
  // a_tile + b_tile: both __shared__ (static), 0 dynamic shared bytes at launch.
  return kernelforge::compute_occupancy(gemm_register_tiled_kernel, kGemmRegBlockThreads, 0);
}

float gemm_register_tiled_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n,
                                    int k) {
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
  gemm_register_tiled_launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_a));
  KF_CUDA_CHECK(cudaFree(d_b));
  KF_CUDA_CHECK(cudaFree(d_c));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
