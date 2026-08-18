#include "attention_fused.cuh"
#include "../cuda_common/error_check.cuh"

#include <cfloat>
#include <cmath>

namespace flashlite {
namespace {

// The single fused attention kernel. Grid: (num_query_tiles, B*H) -- one
// block per (query tile, batch*head) pair. Block: kFusedTileDim threads,
// ONE thread per query row within the tile (thread `threadIdx.x` owns
// query row `blockIdx.x * kFusedTileDim + threadIdx.x`).
//
// Shared memory (dynamic, sized by the launcher below to
// `2 * kFusedTileDim * D * sizeof(float)`): one K tile and one V tile,
// each [kFusedTileDim rows x D cols], reloaded cooperatively by every
// thread in the block once per outer KV-tile iteration -- the same
// cooperative-staging idea V2/V3's kernel 1/3 use, applied to a whole
// [tile x D] row-block at once (no head_dim sub-chunking needed here,
// since D <= kMaxHeadDimFused is small enough to stage a full tile's rows
// in one shot).
__global__ void fused_attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                        const float* __restrict__ v, float* __restrict__ out, int B,
                                        int H, int S, int D, float scale, bool causal) {
  extern __shared__ float smem[];
  float* k_tile = smem;                     // [kFusedTileDim][D]
  float* v_tile = smem + kFusedTileDim * D;  // [kFusedTileDim][D]

  const int bh = blockIdx.y;
  const int h = bh % H;
  const int b = bh / H;

  const int row = blockIdx.x * kFusedTileDim + threadIdx.x;  // this thread's query index i
  const bool row_valid = row < S;

  const long long qkv_base = (static_cast<long long>(b) * H + h) * S * D;

  // Load this thread's own Q row into registers ONCE (D reads total),
  // reused for every key column across every KV tile below -- reading it
  // fresh from global memory per key column (as a naive translation of the
  // per-element recurrence would) would cost an extra O(S*D) global reads
  // per row for no reason, since Q[row, :] never changes within this
  // kernel invocation.
  float q_reg[kMaxHeadDimFused];
  if (row_valid) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      q_reg[d] = q[qkv_base + static_cast<long long>(row) * D + d];
    }
  }

  // Private online-softmax state (docs/online-softmax.md SS2) + output
  // accumulator, one per thread (one per query row) -- this IS the
  // "output accumulated tile-by-tile" the roadmap's Phase 5 deliverable
  // names, with no scores buffer anywhere.
  float m = -FLT_MAX;  // -FLT_MAX sentinel, matching V3's online_softmax_rows_kernel convention
  float l = 0.0f;
  float acc[kMaxHeadDimFused];
#pragma unroll
  for (int d = 0; d < kMaxHeadDimFused; ++d) acc[d] = 0.0f;

  const int num_kv_tiles = (S + kFusedTileDim - 1) / kFusedTileDim;
  for (int t = 0; t < num_kv_tiles; ++t) {
    const int tile_start = t * kFusedTileDim;

    // Cooperative load: this KV tile's rows, staged into shared memory,
    // strided across the block's kFusedTileDim threads (tile has
    // kFusedTileDim*D elements total). Every thread participates
    // regardless of whether ITS OWN row will end up using every loaded
    // column below (causal masking may skip some for this row but not for
    // every row in the block).
    for (int idx = threadIdx.x; idx < kFusedTileDim * D; idx += blockDim.x) {
      const int local_row = idx / D;
      const int local_col = idx % D;
      const int global_row = tile_start + local_row;
      if (global_row < S) {
        k_tile[idx] = k[qkv_base + static_cast<long long>(global_row) * D + local_col];
        v_tile[idx] = v[qkv_base + static_cast<long long>(global_row) * D + local_col];
      } else {
        k_tile[idx] = 0.0f;
        v_tile[idx] = 0.0f;
      }
    }
    __syncthreads();

    if (row_valid && !(causal && tile_start > row)) {
      const int tile_len = min(kFusedTileDim, S - tile_start);
      for (int jj = 0; jj < tile_len; ++jj) {
        const int j = tile_start + jj;
        if (causal && j > row) break;  // j increases with jj -> every later jj in this tile is future too

        float score = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
          score += q_reg[d] * k_tile[jj * D + d];
        }
        score *= scale;

        // The exact per-element online recurrence docs/online-softmax.md
        // SS3/SS5 derives and src/flashlite/online_softmax.py
        // `online_softmax_attention_row` already proves correct on CPU
        // (tests/math/test_online_softmax.py), applied here to a score
        // computed on the fly, one key column at a time, never written to
        // global memory.
        const float m_new = fmaxf(m, score);
        const float correction = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);
        const float p = expf(score - m_new);
#pragma unroll
        for (int d = 0; d < D; ++d) {
          acc[d] = acc[d] * correction + p * v_tile[jj * D + d];
        }
        l = l * correction + p;
        m = m_new;
      }
    }
    __syncthreads();  // before the next iteration overwrites k_tile/v_tile
  }

  if (row_valid) {
    const long long out_row = qkv_base + static_cast<long long>(row) * D;
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;  // l == 0 only for an unsupported fully-masked row
#pragma unroll
    for (int d = 0; d < D; ++d) {
      out[out_row + d] = acc[d] * inv_l;
    }
  }
}

}  // namespace

void launch_attention_fused(const float* q, const float* k, const float* v, float* out, int batch,
                             int heads, int seq_len, int head_dim, bool causal, cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int bh = batch * heads;
  const int num_query_tiles = (seq_len + kFusedTileDim - 1) / kFusedTileDim;

  const dim3 grid(static_cast<unsigned>(num_query_tiles), static_cast<unsigned>(bh));
  const dim3 block(kFusedTileDim);
  const size_t smem_bytes = 2 * static_cast<size_t>(kFusedTileDim) * static_cast<size_t>(head_dim) * sizeof(float);

  fused_attention_kernel<<<grid, block, smem_bytes, stream>>>(q, k, v, out, batch, heads, seq_len,
                                                                head_dim, scale, causal);
  FL_CUDA_CHECK_LAST_ERROR();
}

}  // namespace flashlite
