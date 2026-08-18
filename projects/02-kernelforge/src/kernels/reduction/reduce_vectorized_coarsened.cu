#include "kernels/reduction/reduce_vectorized_coarsened.cuh"

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Cap on the number of blocks this kernel ever launches, regardless of n.
// Each thread grid-strides over multiple float4 groups once n4/block_size
// exceeds this, which is the "coarsening" half of this rung's hypothesis
// (see the .cuh header). 4096 blocks * up to 1024 threads/block is already
// far more parallelism than this GPU's 128 SMs need to be saturated
// (docs/gpu-model.md), so capping below that does not starve occupancy.
constexpr std::size_t kMaxBlocks = 4096;

__global__ void reduce_vectorized_coarsened_kernel(const float* __restrict__ in,
                                                     float* __restrict__ out, std::size_t n) {
  extern __shared__ float sdata[];
  const unsigned tid = threadIdx.x;

  const std::size_t n4 = n / 4;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  const std::size_t idx0 = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  const std::size_t stride4 = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  float local_sum = 0.0f;
  for (std::size_t idx = idx0; idx < n4; idx += stride4) {
    const float4 v = in4[idx];
    local_sum += (v.x + v.y) + (v.z + v.w);
  }

  // Tail: the n % 4 elements not covered by the float4 loop above (at most
  // 3). Handled by the first `tail_count` threads of the flattened grid,
  // which always land in block 0 since tail_count <= 3 < blockDim.x.
  const std::size_t tail_start = n4 * 4;
  const std::size_t tail_count = n - tail_start;
  const std::size_t flat_tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  if (flat_tid < tail_count) {
    local_sum += in[tail_start + flat_tid];
  }

  sdata[tid] = local_sum;
  __syncthreads();

  // Block-reduce tail: copy-identical to reduce_warp_shuffle.cu. The ONLY
  // variable this rung changes vs V3 is the loading phase above.
  for (unsigned stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid < 32) {
    float val = sdata[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (tid == 0) {
      atomicAdd(out, val);
    }
  }
}

} // namespace

void reduce_vectorized_coarsened_launch(const float* d_in, float* d_out, std::size_t n,
                                         int block_size, cudaStream_t stream, dim3& out_grid,
                                         dim3& out_block, bool zero_output) {
  validate_block_size_pow2_1d(block_size, "reduce_vectorized_coarsened_launch");
  if (zero_output) {
    KF_CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
  }

  const dim3 block(static_cast<unsigned>(block_size));
  if (n == 0) {
    out_grid = dim3(0);
    out_block = block;
    return;
  }

  const std::size_t n4 = n / 4;
  std::size_t desired_blocks =
      n4 == 0 ? 1 : (n4 + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  if (desired_blocks > kMaxBlocks) desired_blocks = kMaxBlocks;
  if (desired_blocks < 1) desired_blocks = 1;

  const dim3 grid(static_cast<unsigned>(desired_blocks));
  out_grid = grid;
  out_block = block;

  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  reduce_vectorized_coarsened_kernel<<<grid, block, smem_bytes, stream>>>(d_in, d_out, n);
  KF_CUDA_CHECK_LAST_ERROR();
}

kernelforge::OccupancyReport reduce_vectorized_coarsened_query_occupancy(int block_size) {
  const std::size_t smem_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  return kernelforge::compute_occupancy(reduce_vectorized_coarsened_kernel, block_size,
                                         smem_bytes);
}

float reduce_vectorized_coarsened_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                            int block_size) {
  if (n == 0) {
    *h_sum_out = 0.0f;
    return 0.0f;
  }

  const std::size_t bytes = n * sizeof(float);
  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  reduce_vectorized_coarsened_launch(d_in, d_out, n, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_sum_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
