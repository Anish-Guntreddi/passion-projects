#include "attention_online_softmax.cuh"
#include "../cuda_common/error_check.cuh"

#include <cfloat>
#include <cmath>

namespace flashlite {
namespace {

// Kernel 1: byte-for-byte V2's compute_scores_tiled_kernel
// (attention_tiled.cu) -- duplicated, not shared, per "variants live
// side-by-side" (spec Part 2). See attention_tiled.cuh's header for the
// full cooperative-tile-load design rationale; this file's own header
// explains why only kernel 2 differs for V3.
__global__ void compute_scores_tiled_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                             float* __restrict__ scores, int B, int H, int S, int D,
                                             float scale, bool causal) {
  __shared__ float q_tile[kOnlineSoftmaxAttnTileDim][kOnlineSoftmaxAttnTileDim];
  __shared__ float k_tile[kOnlineSoftmaxAttnTileDim][kOnlineSoftmaxAttnTileDim];

  const int bh = blockIdx.z;
  const int h = bh % H;
  const int b = bh / H;

  const int row = blockIdx.y * kOnlineSoftmaxAttnTileDim + threadIdx.y;  // query index i
  const int col = blockIdx.x * kOnlineSoftmaxAttnTileDim + threadIdx.x;  // key index j
  const int key_row_for_load = blockIdx.x * kOnlineSoftmaxAttnTileDim + threadIdx.y;

  const long long q_base = (static_cast<long long>(b) * H + h) * S * D;
  const long long k_base = q_base;

  float acc = 0.0f;
  const int num_d_tiles = (D + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim;
  for (int t = 0; t < num_d_tiles; ++t) {
    const int d_idx = t * kOnlineSoftmaxAttnTileDim + threadIdx.x;
    q_tile[threadIdx.y][threadIdx.x] =
        (row < S && d_idx < D) ? q[q_base + static_cast<long long>(row) * D + d_idx] : 0.0f;
    k_tile[threadIdx.y][threadIdx.x] = (key_row_for_load < S && d_idx < D)
                                            ? k[k_base + static_cast<long long>(key_row_for_load) * D + d_idx]
                                            : 0.0f;
    __syncthreads();

#pragma unroll
    for (int dd = 0; dd < kOnlineSoftmaxAttnTileDim; ++dd) {
      acc += q_tile[threadIdx.y][dd] * k_tile[threadIdx.x][dd];
    }
    __syncthreads();
  }

  if (row < S && col < S) {
    const long long out_idx = ((static_cast<long long>(b) * H + h) * S + row) * S + col;
    scores[out_idx] = (causal && col > row) ? -INFINITY : acc * scale;
  }
}

// Kernel 2 (V3's ONE changed variable): online-softmax row normalization.
// scores[b,h,i,:] <- softmax(scores[b,h,i,:]) in place, one block per row,
// via the running (m, l) combine rule instead of V1/V2's separate
// max-then-sum passes.
//
// PHASE A (per-thread, one pass): each thread `tid` walks its own strided
// subset of the row (j = tid, tid+block_size, tid+2*block_size, ...),
// maintaining a running (m, l) pair via the exact per-element online
// update (docs/online-softmax.md SS3/SS5, src/flashlite/online_softmax.py
// `combine_online_stats` applied one element at a time: here inlined as
// the scalar recurrence directly, since a single new element IS the
// simplest possible "tile"). This alone reproduces the same identity
// tests/math/test_online_softmax.py already verified on CPU, just
// executed per-thread instead of in a Python loop.
//
// PHASE B (block-level tree reduction): every thread's PRIVATE (m, l) pair
// is itself just another "tile" of the row (a coarser one -- one thread's
// entire strided subset, rather than one element) -- so the SAME combine
// identity used inside Phase A also combines threads' partial results
// together, pairwise, exactly like a standard parallel max/sum reduction,
// except each "add" step is a full (m, l) combine rather than a plain sum.
//
// Sentinel: "no data folded in yet" is represented by m == -FLT_MAX (NOT
// literal -INFINITY), matching V1/V2's softmax_rows_kernel's own existing
// convention (`float local_max = -FLT_MAX;`) -- deliberately avoiding
// IEEE-754 true -inf arithmetic here (`-INFINITY - -INFINITY` is NaN, not
// 0), even though row entries THEMSELVES can genuinely be -INFINITY
// (causal masking writes exactly that in kernel 1 above). This is safe
// because -FLT_MAX is far below any -INFINITY-masked entry's contribution
// threshold: fmaxf(-FLT_MAX, -INFINITY) == -FLT_MAX, so a masked entry
// never accidentally "counts" as real data (src/flashlite/online_softmax.py
// makes the equivalent choice with true -inf, since Python float / float64
// arithmetic handles -inf safely there -- documented as a deliberate,
// float32-hardware-specific difference in docs/online-softmax.md SS10, not
// an inconsistency).
__global__ void online_softmax_rows_kernel(float* __restrict__ scores, int S) {
  extern __shared__ float shared[];
  float* smax = shared;                       // [blockDim.x]
  float* ssum = shared + blockDim.x;           // [blockDim.x]

  float* row = scores + static_cast<long long>(blockIdx.x) * S;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Phase A: per-thread online fold over this thread's strided subset.
  float m = -FLT_MAX;
  float l = 0.0f;
  for (int j = tid; j < S; j += block_size) {
    const float x = row[j];
    const float m_new = fmaxf(m, x);
    const float correction = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);
    l = l * correction + expf(x - m_new);
    m = m_new;
  }
  smax[tid] = m;
  ssum[tid] = l;
  __syncthreads();

  // Phase B: block-level tree reduction, combining pairs of threads'
  // partial (m, l) with the identical combine identity.
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float m_a = smax[tid], l_a = ssum[tid];
      const float m_b = smax[tid + stride], l_b = ssum[tid + stride];
      const float m_new = fmaxf(m_a, m_b);
      float l_new;
      if (m_new == -FLT_MAX) {
        l_new = 0.0f;  // both halves are still the empty identity (S < block_size case)
      } else {
        const float ca = (m_a == -FLT_MAX) ? 0.0f : expf(m_a - m_new);
        const float cb = (m_b == -FLT_MAX) ? 0.0f : expf(m_b - m_new);
        l_new = l_a * ca + l_b * cb;
      }
      smax[tid] = m_new;
      ssum[tid] = l_new;
    }
    __syncthreads();
  }

  const float row_max = smax[0];
  const float row_sum = ssum[0];
  __syncthreads();

  // Final pass: normalize and write (the one pass every variant needs;
  // combined with Phase A above, TWO total passes over the row's global
  // memory, versus V1/V2's three).
  for (int j = tid; j < S; j += block_size) {
    row[j] = expf(row[j] - row_max) / row_sum;
  }
}

// Kernel 3: byte-for-byte V2's weighted_sum_tiled_kernel (attention_tiled.cu).
__global__ void weighted_sum_tiled_kernel(const float* __restrict__ probs, const float* __restrict__ v,
                                           float* __restrict__ out, int B, int H, int S, int D) {
  __shared__ float p_tile[kOnlineSoftmaxAttnTileDim][kOnlineSoftmaxAttnTileDim];
  __shared__ float v_tile[kOnlineSoftmaxAttnTileDim][kOnlineSoftmaxAttnTileDim];

  const int bh = blockIdx.z;
  const int h = bh % H;
  const int b = bh / H;

  const int row = blockIdx.y * kOnlineSoftmaxAttnTileDim + threadIdx.y;
  const int col = blockIdx.x * kOnlineSoftmaxAttnTileDim + threadIdx.x;

  const long long probs_base = (static_cast<long long>(b) * H + h) * S * S;
  const long long v_base = (static_cast<long long>(b) * H + h) * S * D;

  float acc = 0.0f;
  const int num_k_tiles = (S + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim;
  for (int t = 0; t < num_k_tiles; ++t) {
    const int p_col = t * kOnlineSoftmaxAttnTileDim + threadIdx.x;
    const int v_row = t * kOnlineSoftmaxAttnTileDim + threadIdx.y;
    p_tile[threadIdx.y][threadIdx.x] =
        (row < S && p_col < S) ? probs[probs_base + static_cast<long long>(row) * S + p_col] : 0.0f;
    v_tile[threadIdx.y][threadIdx.x] =
        (v_row < S && col < D) ? v[v_base + static_cast<long long>(v_row) * D + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kOnlineSoftmaxAttnTileDim; ++kk) {
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

void launch_attention_online_softmax(const float* q, const float* k, const float* v, float* out,
                                      float* scores_scratch, int batch, int heads, int seq_len,
                                      int head_dim, bool causal, cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int bh = batch * heads;

  const dim3 score_block(kOnlineSoftmaxAttnTileDim, kOnlineSoftmaxAttnTileDim);
  const dim3 score_grid(static_cast<unsigned>((seq_len + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim),
                         static_cast<unsigned>((seq_len + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim),
                         static_cast<unsigned>(bh));
  compute_scores_tiled_kernel<<<score_grid, score_block, 0, stream>>>(
      q, k, scores_scratch, batch, heads, seq_len, head_dim, scale, causal);
  FL_CUDA_CHECK_LAST_ERROR();

  const long long row_count = static_cast<long long>(bh) * seq_len;
  const size_t online_softmax_shared_bytes = 2 * static_cast<size_t>(kOnlineSoftmaxBlockSize) * sizeof(float);
  online_softmax_rows_kernel<<<static_cast<unsigned int>(row_count), kOnlineSoftmaxBlockSize,
                                online_softmax_shared_bytes, stream>>>(scores_scratch, seq_len);
  FL_CUDA_CHECK_LAST_ERROR();

  const dim3 out_block(kOnlineSoftmaxAttnTileDim, kOnlineSoftmaxAttnTileDim);
  const dim3 out_grid(static_cast<unsigned>((head_dim + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim),
                       static_cast<unsigned>((seq_len + kOnlineSoftmaxAttnTileDim - 1) / kOnlineSoftmaxAttnTileDim),
                       static_cast<unsigned>(bh));
  weighted_sum_tiled_kernel<<<out_grid, out_block, 0, stream>>>(scores_scratch, v, out, batch, heads,
                                                                 seq_len, head_dim);
  FL_CUDA_CHECK_LAST_ERROR();
}

}  // namespace flashlite
