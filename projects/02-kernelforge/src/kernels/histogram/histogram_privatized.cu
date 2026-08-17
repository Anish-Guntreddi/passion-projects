#include "kernels/histogram/histogram_privatized.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"
#include "kernels/histogram/histogram_common.cuh"

namespace kernelforge::kernels {

namespace {

__global__ void histogram_privatized_kernel(const float* __restrict__ in,
                                              int* __restrict__ hist, std::size_t n,
                                              int num_bins) {
  extern __shared__ int block_hist[];
  for (int b = static_cast<int>(threadIdx.x); b < num_bins; b += static_cast<int>(blockDim.x)) {
    block_hist[b] = 0;
  }
  __syncthreads();

  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const int bin = histogram_bin_of(in[i], num_bins);
    atomicAdd(&block_hist[bin], 1);
  }
  __syncthreads();

  // Merge this block's private histogram into the global one: exactly
  // num_bins global atomics per block (the ONE variable changed vs V1 --
  // see header comment), instead of one global atomic per element.
  for (int b = static_cast<int>(threadIdx.x); b < num_bins; b += static_cast<int>(blockDim.x)) {
    const int count = block_hist[b];
    if (count != 0) {
      atomicAdd(&hist[b], count);
    }
  }
}

} // namespace

void histogram_privatized_launch(const float* d_in, int* d_hist, std::size_t n, int num_bins,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block, bool zero_output) {
  validate_block_size_1d(block_size, "histogram_privatized_launch");
  validate_num_bins_fits_shared_mem(num_bins, "histogram_privatized_launch");
  validate_n_fits_int_bin_counts(n, "histogram_privatized_launch");
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

  const std::size_t smem_bytes = static_cast<std::size_t>(num_bins) * sizeof(int);
  histogram_privatized_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_hist, n, num_bins);
  KF_CUDA_CHECK_LAST_ERROR();
}

float histogram_privatized_run_host(const float* h_in, std::size_t n, int* h_hist_out,
                                     int num_bins, int block_size) {
  validate_num_bins_fits_shared_mem(num_bins, "histogram_privatized_run_host");
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
  histogram_privatized_launch(d_in, d_hist, n, num_bins, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_hist_out, d_hist, hist_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_hist));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
