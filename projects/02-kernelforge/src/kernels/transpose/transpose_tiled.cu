#include "kernels/transpose/transpose_tiled.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
  // Deliberately unpadded: [TILE][TILE], not [TILE][TILE+1]. See
  // transpose_tiled.cuh for why (isolates "global coalescing" as the one
  // variable changed from the naive kernel; padding is future work).
  __shared__ float tile[kTransposeTileDim][kTransposeTileDim];

  const int col0 = blockIdx.x * kTransposeTileDim; // tile origin in input-column space
  const int row0 = blockIdx.y * kTransposeTileDim; // tile origin in input-row space

  // Stage 1: read a TILE x TILE block of `in` into shared memory.
  // Consecutive threadIdx.x -> consecutive input column -> coalesced read.
  const int in_row = row0 + threadIdx.y;
  const int in_col = col0 + threadIdx.x;
  if (in_row < rows && in_col < cols) {
    tile[threadIdx.y][threadIdx.x] = in[static_cast<std::size_t>(in_row) * cols + in_col];
  }
  __syncthreads();

  // Stage 2: write the transposed tile out. Output tile origin is the
  // input tile origin with row/col swapped. Consecutive threadIdx.x now
  // maps to consecutive output column -> coalesced write.
  const int out_row = col0 + threadIdx.y; // in [0, cols) -- output has `cols` rows
  const int out_col = row0 + threadIdx.x; // in [0, rows) -- output has `rows` columns
  if (out_row < cols && out_col < rows) {
    // tile[threadIdx.x][threadIdx.y]: the transposed read out of shared
    // memory. This is where the (unaddressed) bank conflict lives.
    out[static_cast<std::size_t>(out_row) * rows + out_col] = tile[threadIdx.x][threadIdx.y];
  }
}

} // namespace

void transpose_tiled_launch(const float* d_in, float* d_out, int rows, int cols,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kTransposeTileDim, kTransposeTileDim);
  const dim3 grid(static_cast<unsigned>((cols + kTransposeTileDim - 1) / kTransposeTileDim),
                   static_cast<unsigned>((rows + kTransposeTileDim - 1) / kTransposeTileDim));
  out_grid = grid;
  out_block = block;
  transpose_tiled_kernel<<<grid, block, 0, stream>>>(d_in, d_out, rows, cols);
  KF_CUDA_CHECK_LAST_ERROR();
}

kernelforge::OccupancyReport transpose_tiled_query_occupancy() {
  return kernelforge::compute_occupancy(transpose_tiled_kernel,
                                         kTransposeTileDim * kTransposeTileDim, 0);
}

float transpose_tiled_run_host(const float* h_in, float* h_out, int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (n == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.
  const std::size_t bytes = n * sizeof(float);

  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  transpose_tiled_launch(d_in, d_out, rows, cols, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
