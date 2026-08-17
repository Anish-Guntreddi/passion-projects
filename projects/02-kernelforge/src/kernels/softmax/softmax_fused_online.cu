#include "kernels/softmax/softmax_fused_online.cuh"

#include <cfloat>

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Online-softmax pairwise combine: (m, l) <- combine((m, l), (other_m,
// other_l)), in place. Mathematically exact rescale (see this file's
// .cuh header), with one explicit guard: if BOTH sides are the "empty"
// identity (m == other_m == -FLT_MAX, meaning neither side has visited
// any element yet -- only possible when cols < a lane's/warp's share of
// the block), `m - m_new` and `other_m - m_new` are both `-FLT_MAX -
// (-FLT_MAX)` = 0 exactly (not -inf, since this file uses -FLT_MAX, not
// -INFINITY, as the identity -- see combine's call sites), so expf(0)=1
// and l_new = l*1 + other_l*1 = 0 + 0 = 0 with NO NaN risk. -FLT_MAX
// (not -INFINITY) is used as the identity specifically so this combine
// never has to special-case an actual IEEE -inf/-inf subtraction.
__device__ __forceinline__ void online_combine(float& m, float& l, float other_m, float other_l) {
  const float m_new = fmaxf(m, other_m);
  l = l * expf(m - m_new) + other_l * expf(other_m - m_new);
  m = m_new;
}

// Block-wide online-combine reduction: every thread passes its own local
// (m, l) partial; every thread gets back the SAME final block-wide (m, l).
// Same two-level warp-shuffle shape as softmax_warp_shuffle.cu's
// block_reduce_max/block_reduce_sum (ADR 0004: full mask, `_sync`
// family), generalized to reduce a PAIR via online_combine instead of a
// single float via fmaxf/`+` -- see this file's .cuh header for why that
// generalization is valid.
__device__ void block_combine_online(float& m, float& l) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_m = __shfl_down_sync(0xffffffffu, m, offset);
    const float other_l = __shfl_down_sync(0xffffffffu, l, offset);
    online_combine(m, l, other_m, other_l);
  }
  __shared__ float warp_m[32];
  __shared__ float warp_l[32];
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp_id = threadIdx.x >> 5;
  if (lane == 0) {
    warp_m[warp_id] = m;
    warp_l[warp_id] = l;
  }
  __syncthreads();

  const unsigned num_warps = (blockDim.x + 31) / 32;
  if (threadIdx.x < num_warps) {
    m = warp_m[threadIdx.x];
    l = warp_l[threadIdx.x];
  } else {
    m = -FLT_MAX;
    l = 0.0f;
  }
  if (warp_id == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_m = __shfl_down_sync(0xffffffffu, m, offset);
      const float other_l = __shfl_down_sync(0xffffffffu, l, offset);
      online_combine(m, l, other_m, other_l);
    }
  }
  __shared__ float result_m;
  __shared__ float result_l;
  if (threadIdx.x == 0) {
    result_m = m;
    result_l = l;
  }
  __syncthreads();
  m = result_m;
  l = result_l;
}

__global__ void softmax_fused_online_kernel(const float* __restrict__ in, float* __restrict__ out,
                                             int cols) {
  const int row = static_cast<int>(blockIdx.x);
  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);

  // --- Pass 1: fused online max + exp-sum (ONE read of the row) ---
  float m = -FLT_MAX;
  float l = 0.0f;
  for (int c = tid; c < cols; c += stride) {
    online_combine(m, l, row_in[c], 1.0f); // fold element x in as the pair (x, exp(x-x)=1)
  }
  block_combine_online(m, l);
  const float row_max = m;
  const float row_sum = l;

  // --- Pass 2: normalize and write (second, final read of the row) ---
  for (int c = tid; c < cols; c += stride) row_out[c] = expf(row_in[c] - row_max) / row_sum;
}

} // namespace

void softmax_fused_online_launch(const float* d_in, float* d_out, int rows, int cols,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "softmax_fused_online_launch");
  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(static_cast<unsigned>(block_size));
  out_grid = grid;
  out_block = block;
  if (rows == 0) return;

  softmax_fused_online_kernel<<<grid, block, 0, stream>>>(d_in, d_out, cols);
  KF_CUDA_CHECK_LAST_ERROR();
}

float softmax_fused_online_run_host(const float* h_in, float* h_out, int rows, int cols,
                                     int block_size) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (rows == 0 || cols == 0) return 0.0f;
  const std::size_t bytes = n * sizeof(float);

  float* d_in = nullptr;
  float* d_out = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
  KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
  KF_CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  softmax_fused_online_launch(d_in, d_out, rows, cols, block_size, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_in));
  KF_CUDA_CHECK(cudaFree(d_out));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
