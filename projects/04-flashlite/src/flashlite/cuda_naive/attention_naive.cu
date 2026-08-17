#include "attention_naive.cuh"
#include "error_check.cuh"

#include <cfloat>
#include <cmath>

namespace flashlite {
namespace {

// Kernel 1 (score materialization) / kernel 3 (weighted sum): one thread
// per output element, plain grid-stride-free 1D launch (total <= 2^31-1
// covers every shape this repo's tests/benchmarks use).
constexpr int kElementwiseBlockSize = 256;
// Kernel 2 (row softmax): one block per (b, h, i) row. Must be a power of
// two -- the halving shared-memory reduction below assumes it.
constexpr int kSoftmaxBlockSize = 256;

// out[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) * scale, or -inf where
// causal && j > i. Naive: no shared-memory reuse of q/k rows across
// threads, full D-length loop per thread -- this is what V2 (tiled) exists
// to fix (docs/attention-math.md's IO-cost discussion).
__global__ void compute_scores_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                       float* __restrict__ scores, int B, int H, int S, int D,
                                       float scale, bool causal) {
  const long long total = static_cast<long long>(B) * H * S * S;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  long long rem = idx;
  const int j = static_cast<int>(rem % S);
  rem /= S;
  const int i = static_cast<int>(rem % S);
  rem /= S;
  const int h = static_cast<int>(rem % H);
  rem /= H;
  const int b = static_cast<int>(rem);

  if (causal && j > i) {
    scores[idx] = -INFINITY;
    return;
  }

  const long long q_row = ((static_cast<long long>(b) * H + h) * S + i) * D;
  const long long k_row = ((static_cast<long long>(b) * H + h) * S + j) * D;

  float acc = 0.0f;
  for (int d = 0; d < D; ++d) {
    acc += q[q_row + d] * k[k_row + d];
  }
  scores[idx] = acc * scale;
}

// In-place row softmax: scores[b,h,i,:] <- softmax(scores[b,h,i,:]).
// Ordinary two-pass, numerically-stable softmax (subtract row max, then
// normalize) -- NOT the online/streaming running-max variant Phase 4
// introduces; every row is already fully materialized in global memory by
// kernel 1, so there is nothing to stream across tiles yet.
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

// out[b,h,i,d] = sum_j probs[b,h,i,j] * v[b,h,j,d]. Naive: one thread per
// output element, full S-length loop per thread.
__global__ void weighted_sum_kernel(const float* __restrict__ probs, const float* __restrict__ v,
                                     float* __restrict__ out, int B, int H, int S, int D) {
  const long long total = static_cast<long long>(B) * H * S * D;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  long long rem = idx;
  const int d = static_cast<int>(rem % D);
  rem /= D;
  const int i = static_cast<int>(rem % S);
  rem /= S;
  const int h = static_cast<int>(rem % H);
  rem /= H;
  const int b = static_cast<int>(rem);

  const long long probs_row = ((static_cast<long long>(b) * H + h) * S + i) * S;
  const long long v_base = (static_cast<long long>(b) * H + h) * S * D;

  float acc = 0.0f;
  for (int j = 0; j < S; ++j) {
    acc += probs[probs_row + j] * v[v_base + static_cast<long long>(j) * D + d];
  }
  out[idx] = acc;
}

}  // namespace

void launch_attention_naive(const float* q, const float* k, const float* v, float* out,
                             float* scores_scratch, int batch, int heads, int seq_len,
                             int head_dim, bool causal, cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  const long long score_total = static_cast<long long>(batch) * heads * seq_len * seq_len;
  const long long score_blocks = (score_total + kElementwiseBlockSize - 1) / kElementwiseBlockSize;
  compute_scores_kernel<<<static_cast<unsigned int>(score_blocks), kElementwiseBlockSize, 0, stream>>>(
      q, k, scores_scratch, batch, heads, seq_len, head_dim, scale, causal);
  FL_CUDA_CHECK_LAST_ERROR();

  const long long row_count = static_cast<long long>(batch) * heads * seq_len;
  softmax_rows_kernel<<<static_cast<unsigned int>(row_count), kSoftmaxBlockSize,
                         kSoftmaxBlockSize * sizeof(float), stream>>>(scores_scratch, seq_len);
  FL_CUDA_CHECK_LAST_ERROR();

  const long long out_total = static_cast<long long>(batch) * heads * seq_len * head_dim;
  const long long out_blocks = (out_total + kElementwiseBlockSize - 1) / kElementwiseBlockSize;
  weighted_sum_kernel<<<static_cast<unsigned int>(out_blocks), kElementwiseBlockSize, 0, stream>>>(
      scores_scratch, v, out, batch, heads, seq_len, head_dim);
  FL_CUDA_CHECK_LAST_ERROR();
}

}  // namespace flashlite
