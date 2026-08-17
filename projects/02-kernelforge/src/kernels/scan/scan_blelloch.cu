#include "kernels/scan/scan_blelloch.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

__global__ void scan_blelloch_block_kernel(const float* __restrict__ in, float* __restrict__ out,
                                            float* __restrict__ block_sums, std::size_t n) {
  extern __shared__ float sdata[];
  const unsigned tid = threadIdx.x;
  const unsigned bsize = blockDim.x;
  const std::size_t gidx = static_cast<std::size_t>(blockIdx.x) * bsize + tid;

  const float orig = (gidx < n) ? in[gidx] : 0.0f;
  sdata[tid] = orig;
  __syncthreads();

  // Up-sweep (reduce) phase: build a partial-sums tree in place.
  for (unsigned stride = 1; stride < bsize; stride <<= 1) {
    const unsigned idx = (tid + 1) * stride * 2 - 1;
    if (idx < bsize) {
      sdata[idx] += sdata[idx - stride];
    }
    __syncthreads();
  }

  __shared__ float block_total;
  if (tid == 0) {
    block_total = sdata[bsize - 1];
    sdata[bsize - 1] = 0.0f; // identity, to seed the down-sweep -> exclusive scan.
  }
  __syncthreads();

  // Down-sweep (distribute) phase.
  for (unsigned stride = bsize / 2; stride >= 1; stride >>= 1) {
    const unsigned idx = (tid + 1) * stride * 2 - 1;
    if (idx < bsize) {
      const float t = sdata[idx - stride];
      sdata[idx - stride] = sdata[idx];
      sdata[idx] += t;
    }
    __syncthreads();
  }

  // sdata[tid] now holds the EXCLUSIVE scan; add the original value back
  // to produce this kernel's INCLUSIVE output (see .cuh header).
  const float inclusive = sdata[tid] + orig;
  if (gidx < n) {
    out[gidx] = inclusive;
  }
  if (tid == bsize - 1) {
    block_sums[blockIdx.x] = block_total;
  }
}

} // namespace

void scan_blelloch_block_pass_launch(const float* d_in, float* d_out, float* d_block_sums,
                                      std::size_t n, int block_size, cudaStream_t stream) {
  validate_block_size_pow2_1d(block_size, "scan_blelloch_block_pass_launch");
  const dim3 block(static_cast<unsigned>(block_size));
  const std::size_t num_blocks =
      (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  const dim3 grid(static_cast<unsigned>(num_blocks));
  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  scan_blelloch_block_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, d_block_sums, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

void scan_blelloch_launch(const float* d_in, float* d_out, const ScanScratch& scratch,
                           std::size_t n, int block_size, cudaStream_t stream, dim3& out_grid,
                           dim3& out_block) {
  scan_multilevel_launch(&scan_blelloch_block_pass_launch, d_in, d_out, scratch, n, block_size,
                          stream, out_grid, out_block);
}

float scan_blelloch_run_host(const float* h_in, float* h_out, std::size_t n, int block_size) {
  if (n == 0) return 0.0f;

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
  scan_blelloch_launch(d_in, d_out, scratch, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  scan_free_scratch(scratch);
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
