#include "kernels/histogram/histogram_global_atomic.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"
#include "kernels/histogram/histogram_common.cuh"

namespace kernelforge::kernels {

namespace {

__global__ void histogram_global_atomic_kernel(const float* __restrict__ in,
                                                 int* __restrict__ hist, std::size_t n,
                                                 int num_bins) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const int bin = histogram_bin_of(in[i], num_bins);
    atomicAdd(&hist[bin], 1);
  }
}

} // namespace

void histogram_global_atomic_launch(const float* d_in, int* d_hist, std::size_t n, int num_bins,
                                     int block_size, cudaStream_t stream, dim3& out_grid,
                                     dim3& out_block, bool zero_output) {
  validate_block_size_1d(block_size, "histogram_global_atomic_launch");
  if (num_bins <= 0) {
    throw std::invalid_argument("histogram_global_atomic_launch: num_bins must be > 0");
  }
  validate_n_fits_int_bin_counts(n, "histogram_global_atomic_launch");
  if (zero_output) {
    KF_CUDA_CHECK(cudaMemsetAsync(d_hist, 0, static_cast<std::size_t>(num_bins) * sizeof(int), stream));
  }

  const std::size_t num_blocks =
      n == 0 ? 0 : (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  const dim3 grid(static_cast<unsigned>(num_blocks));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (num_blocks == 0) return;

  histogram_global_atomic_kernel<<<grid, block, 0, stream>>>(d_in, d_hist, n, num_bins);
  KF_CUDA_CHECK_LAST_ERROR();
}

float histogram_global_atomic_run_host(const float* h_in, std::size_t n, int* h_hist_out,
                                        int num_bins, int block_size) {
  if (num_bins <= 0) {
    throw std::invalid_argument("histogram_global_atomic_run_host: num_bins must be > 0");
  }
  if (n == 0) {
    for (int b = 0; b < num_bins; ++b) h_hist_out[b] = 0;
    return 0.0f;
  }

  const std::size_t bytes = n * sizeof(float);
  const std::size_t hist_bytes = static_cast<std::size_t>(num_bins) * sizeof(int);
  float* d_in = nullptr;
  int* d_hist = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_hist, hist_bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  histogram_global_atomic_launch(d_in, d_hist, n, num_bins, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_hist_out, d_hist, hist_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_hist));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
