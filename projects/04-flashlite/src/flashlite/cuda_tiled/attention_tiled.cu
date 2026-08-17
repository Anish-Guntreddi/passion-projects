#include "attention_tiled.cuh"
#include "../cuda_common/error_check.cuh"

#include <cfloat>
#include <cmath>

namespace flashlite {
namespace {

// Kernel 2 (row softmax): one block per (b, h, i) row. Must be a power of
// two -- the halving shared-memory reduction below assumes it. Unrelated to
// kAttnTileDim (kernel 1/3's tile dimension) on purpose: this kernel's
// parallelism is "threads cooperating over one row of length S", not
// "threads covering a 2-D output tile", so there is no reason the two
// should share a launch-config constant.
constexpr int kSoftmaxBlockSize = 256;

// out[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) * scale, or -inf where
// causal && j > i. TILED: one kAttnTileDim x kAttnTileDim thread block
// computes one kAttnTileDim x kAttnTileDim tile of one (b,h) slice's score
// matrix. Q and K sub-tiles are cooperatively loaded into shared memory
// once per head_dim chunk and reused by every thread in the block for that
// chunk (the "one variable" this rung changes vs. V1's compute_scores_kernel
// -- see attention_tiled.cuh's header for the full hypothesis).
//
// Load layout note: q_tile[ty][tx] = Q[row(ty), d_chunk+tx] and
// k_tile[ty][tx] = K[col(ty), d_chunk+tx] -- BOTH loads index shared memory
// by (row/col-within-tile, d-within-chunk), matching Q/K's actual
// [.., S, D] memory layout (D contiguous) so both loads are coalesced
// (consecutive threadIdx.x -> consecutive D address). The compute step then
// reads k_tile "transposed" (k_tile[threadIdx.x][dd] instead of
// k_tile[threadIdx.y][dd]) to get K[col, d] for THIS thread's own output
// column -- this is what makes Q @ K^T work from two identically-laid-out
// tiles without a separate transpose pass.
__global__ void compute_scores_tiled_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                             float* __restrict__ scores, int B, int H, int S, int D,
                                             float scale, bool causal) {
  __shared__ float q_tile[kAttnTileDim][kAttnTileDim];
  __shared__ float k_tile[kAttnTileDim][kAttnTileDim];

  const int bh = blockIdx.z;
  const int h = bh % H;
  const int b = bh / H;

  const int row = blockIdx.y * kAttnTileDim + threadIdx.y;  // query index i
  const int col = blockIdx.x * kAttnTileDim + threadIdx.x;  // key index j
  const int key_row_for_load = blockIdx.x * kAttnTileDim + threadIdx.y;  // K row this thread loads

  const long long q_base = (static_cast<long long>(b) * H + h) * S * D;
  const long long k_base = q_base;  // q and k share [B, H, S, D] shape

  float acc = 0.0f;
  const int num_d_tiles = (D + kAttnTileDim - 1) / kAttnTileDim;
  for (int t = 0; t < num_d_tiles; ++t) {
    const int d_idx = t * kAttnTileDim + threadIdx.x;
    q_tile[threadIdx.y][threadIdx.x] =
        (row < S && d_idx < D) ? q[q_base + static_cast<long long>(row) * D + d_idx] : 0.0f;
    k_tile[threadIdx.y][threadIdx.x] = (key_row_for_load < S && d_idx < D)
                                            ? k[k_base + static_cast<long long>(key_row_for_load) * D + d_idx]
                                            : 0.0f;
    __syncthreads();

#pragma unroll
    for (int dd = 0; dd < kAttnTileDim; ++dd) {
      acc += q_tile[threadIdx.y][dd] * k_tile[threadIdx.x][dd];
    }
    __syncthreads();
  }

  if (row < S && col < S) {
    const long long out_idx = ((static_cast<long long>(b) * H + h) * S + row) * S + col;
    scores[out_idx] = (causal && col > row) ? -INFINITY : acc * scale;
  }
}

// In-place row softmax: scores[b,h,i,:] <- softmax(scores[b,h,i,:]).
// Byte-for-byte identical to V1's softmax_rows_kernel (attention_naive.cu)
// -- deliberately NOT tiled or changed in any way. "Conceptually simple
// softmax retained" (roadmap Phase 3) is this repo's literal wording for
// why: every row is already fully materialized in global memory by kernel
// 1, so there is nothing about THIS kernel that tiling would change; the
// online/streaming softmax that removes the need for a fully materialized
// row is Phase 4's variable, not Phase 3's.
__global__ void softmax_rows_kernel(float* __restrict__ scores, int S) {
  extern __shared__ float shared[];
  float* row = scores + static_cast<long long>(blockIdx.x) * S;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  float local_max = -FLT_MAX;
  for (int j = tid; j < S; j += block_size) {
    local_max = fmaxf(local_max, row[j]);
  }
  shared[tid] = local_max;
  __syncthreads();
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    __syncthreads();
  }
  const float row_max = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int j = tid; j < S; j += block_size) {
    local_sum += expf(row[j] - row_max);
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float row_sum = shared[0];
  __syncthreads();

  for (int j = tid; j < S; j += block_size) {
    row[j] = expf(row[j] - row_max) / row_sum;
  }
}

// out[b,h,i,d] = sum_j probs[b,h,i,j] * v[b,h,j,d]. TILED: a plain,
// non-transposed GEMM shape (P [S, S] @ V [S, D] = O [S, D]) -- the same
// cooperative-tile-load pattern as kernel 1, without the k_tile
// "transpose" wrinkle (P and V's reduction/output axes already line up
// with each other's memory layout the way KernelForge's gemm_tiled_kernel
// expects: p_tile loads P's row-major [S(row), S(reduction)] layout,
// v_tile loads V's row-major [S(reduction), D(col)] layout, directly
// analogous to A[m,k] @ B[k,n]).
__global__ void weighted_sum_tiled_kernel(const float* __restrict__ probs, const float* __restrict__ v,
                                           float* __restrict__ out, int B, int H, int S, int D) {
  __shared__ float p_tile[kAttnTileDim][kAttnTileDim];
  __shared__ float v_tile[kAttnTileDim][kAttnTileDim];

  const int bh = blockIdx.z;
  const int h = bh % H;
  const int b = bh / H;

  const int row = blockIdx.y * kAttnTileDim + threadIdx.y;  // query index i
  const int col = blockIdx.x * kAttnTileDim + threadIdx.x;  // head_dim index d

  const long long probs_base = (static_cast<long long>(b) * H + h) * S * S;
  const long long v_base = (static_cast<long long>(b) * H + h) * S * D;

  float acc = 0.0f;
  const int num_k_tiles = (S + kAttnTileDim - 1) / kAttnTileDim;
  for (int t = 0; t < num_k_tiles; ++t) {
    const int p_col = t * kAttnTileDim + threadIdx.x;  // key index j, for P's load
    const int v_row = t * kAttnTileDim + threadIdx.y;  // key index j, for V's load
    p_tile[threadIdx.y][threadIdx.x] =
        (row < S && p_col < S) ? probs[probs_base + static_cast<long long>(row) * S + p_col] : 0.0f;
    v_tile[threadIdx.y][threadIdx.x] =
        (v_row < S && col < D) ? v[v_base + static_cast<long long>(v_row) * D + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kAttnTileDim; ++kk) {
      acc += p_tile[threadIdx.y][kk] * v_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < S && col < D) {
    const long long out_idx = ((static_cast<long long>(b) * H + h) * S + row) * D + col;
    out[out_idx] = acc;
  }
}

}  // namespace

void launch_attention_tiled(const float* q, const float* k, const float* v, float* out,
                             float* scores_scratch, int batch, int heads, int seq_len, int head_dim,
                             bool causal, cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int bh = batch * heads;

  const dim3 score_block(kAttnTileDim, kAttnTileDim);
  const dim3 score_grid(static_cast<unsigned>((seq_len + kAttnTileDim - 1) / kAttnTileDim),
                         static_cast<unsigned>((seq_len + kAttnTileDim - 1) / kAttnTileDim),
                         static_cast<unsigned>(bh));
  compute_scores_tiled_kernel<<<score_grid, score_block, 0, stream>>>(
      q, k, scores_scratch, batch, heads, seq_len, head_dim, scale, causal);
  FL_CUDA_CHECK_LAST_ERROR();

  const long long row_count = static_cast<long long>(bh) * seq_len;
  softmax_rows_kernel<<<static_cast<unsigned int>(row_count), kSoftmaxBlockSize,
                         kSoftmaxBlockSize * sizeof(float), stream>>>(scores_scratch, seq_len);
  FL_CUDA_CHECK_LAST_ERROR();

  const dim3 out_block(kAttnTileDim, kAttnTileDim);
  const dim3 out_grid(static_cast<unsigned>((head_dim + kAttnTileDim - 1) / kAttnTileDim),
                       static_cast<unsigned>((seq_len + kAttnTileDim - 1) / kAttnTileDim),
                       static_cast<unsigned>(bh));
  weighted_sum_tiled_kernel<<<out_grid, out_block, 0, stream>>>(scores_scratch, v, out, batch, heads,
                                                                 seq_len, head_dim);
  FL_CUDA_CHECK_LAST_ERROR();
}

}  // namespace flashlite
