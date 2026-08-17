#include "kernels/scan/scan_hillis_steele.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void scan_hillis_steele_block_kernel(const float* __restrict__ in,
                                                  float* __restrict__ out,
                                                  float* __restrict__ block_sums, std::size_t n) {
  extern __shared__ float smem[];
  float* buf0 = smem;
  float* buf1 = smem + blockDim.x;

  const unsigned tid = threadIdx.x;
  const std::size_t gidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  buf0[tid] = (gidx < n) ? in[gidx] : 0.0f;
  __syncthreads();

  float* src = buf0;
  float* dst = buf1;
  for (unsigned offset = 1; offset < blockDim.x; offset <<= 1) {
    const float added = (tid >= offset) ? (src[tid] + src[tid - offset]) : src[tid];
    dst[tid] = added;
    __syncthreads();
    float* tmp = src;
    src = dst;
    dst = tmp;
  }

  if (gidx < n) {
    out[gidx] = src[tid];
  }
  if (tid == blockDim.x - 1) {
    // Padding beyond n within this (possibly partial, last) block
    // contributed 0, so src[tid] here is exactly this block's real-element
    // total regardless of whether the block is full.
    block_sums[blockIdx.x] = src[tid];
  }
}

} // namespace

void scan_hillis_steele_block_pass_launch(const float* d_in, float* d_out, float* d_block_sums,
                                           std::size_t n, int block_size, cudaStream_t stream) {
  validate_block_size_pow2_1d(block_size, "scan_hillis_steele_block_pass_launch");
  const dim3 block(static_cast<unsigned>(block_size));
  const std::size_t num_blocks =
      (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  const dim3 grid(static_cast<unsigned>(num_blocks));
  const std::size_t smem_bytes = 2ull * static_cast<std::size_t>(block_size) * sizeof(float);
  scan_hillis_steele_block_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, d_block_sums,
                                                                        n);
  KF_CUDA_CHECK_LAST_ERROR();
}

void scan_hillis_steele_launch(const float* d_in, float* d_out, const ScanScratch& scratch,
                                std::size_t n, int block_size, cudaStream_t stream,
                                dim3& out_grid, dim3& out_block) {
  scan_multilevel_launch(&scan_hillis_steele_block_pass_launch, d_in, d_out, scratch, n,
                          block_size, stream, out_grid, out_block);
}

float scan_hillis_steele_run_host(const float* h_in, float* h_out, std::size_t n,
                                   int block_size) {
  if (n == 0) return 0.0f; // FR6 edge size: legal, nothing to allocate/launch.

  const std::size_t bytes = n * sizeof(float);
  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  ScanScratch scratch = scan_alloc_scratch(block_size);

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  scan_hillis_steele_launch(d_in, d_out, scratch, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  scan_free_scratch(scratch);
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
