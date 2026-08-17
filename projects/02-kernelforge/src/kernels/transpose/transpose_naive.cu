#include "kernels/transpose/transpose_naive.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

constexpr int kNaiveBlockDim = 32; // 32x32 = 1024 threads/block, matches transpose_tiled's
                                    // tile size so the two kernels differ in exactly one
                                    // variable: whether shared memory stages the transpose.

__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    // Read: consecutive threadIdx.x -> consecutive `col` -> contiguous address. Coalesced.
    // Write: consecutive threadIdx.x -> consecutive `col` -> address stride = rows floats.
    // Pathological: one memory transaction per thread instead of one per warp.
    out[static_cast<std::size_t>(col) * rows + row] =
        in[static_cast<std::size_t>(row) * cols + col];
  }
}

} // namespace

void transpose_naive_launch(const float* d_in, float* d_out, int rows, int cols,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  const dim3 block(kNaiveBlockDim, kNaiveBlockDim);
  const dim3 grid(static_cast<unsigned>((cols + kNaiveBlockDim - 1) / kNaiveBlockDim),
                   static_cast<unsigned>((rows + kNaiveBlockDim - 1) / kNaiveBlockDim));
  out_grid = grid;
  out_block = block;
  transpose_naive_kernel<<<grid, block, 0, stream>>>(d_in, d_out, rows, cols);
  KF_CUDA_CHECK_LAST_ERROR();
}

float transpose_naive_run_host(const float* h_in, float* h_out, int rows, int cols) {
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
  transpose_naive_launch(d_in, d_out, rows, cols, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
