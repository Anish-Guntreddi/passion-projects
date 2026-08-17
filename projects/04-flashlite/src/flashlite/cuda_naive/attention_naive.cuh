// V1: naive custom attention kernels (spec ladder SS1.3: "Naive custom
// kernel materializing scores/output intermediates").
//
// Mathematical invariant (docs/attention-math.md):
//     S = (Q K^T) / sqrt(d);  P = softmax(S);  O = P V
// checked against flashlite.reference.attention (V0) elementwise within
// tolerance (tests/correctness/test_naive_attention.py).
//
// Tensor shape/layout contract (ADR 0002, enforced by bindings.cpp before
// any kernel launches): q, k, v are [B, H, S, D] contiguous float32 CUDA
// tensors, all three the SAME shape (self-attention MVP). `causal` masks
// key position j > query position i with -inf before softmax (ADR 0003).
//
// Design: this is deliberately the simplest correct implementation, not a
// tuned one -- three separate kernels, one thread per output element in
// kernels 1 and 3 (a naive full loop over head_dim / seq_len respectively,
// no shared-memory reuse of Q/K/V tiles), one block per row in kernel 2. It
// *materializes* the full [B, H, S, S] score/probability matrix in global
// memory between kernels, exactly matching the V1 rung's definition and
// setting up the contrast the V2 (tiled) / V4 (fused) variants exist to
// fix (spec SS1.3, SS1.9: "optimizations justified by memory/compute
// reasoning" -- Phase 2 will quantify exactly how wasteful this
// materialization is before Phase 3 removes it).
#pragma once

#include <cuda_runtime.h>

namespace flashlite {

// Launches the 3-kernel naive attention pipeline on `stream`:
//   1. compute_scores_kernel:  scores[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) * scale,
//                               masked to -inf where causal && j > i.
//   2. softmax_rows_kernel:    scores[b,h,i,:] <- softmax(scores[b,h,i,:]) in place
//                               (two-pass, block-per-row, shared-memory max/sum
//                               reduction -- ordinary numerically-stable softmax,
//                               NOT the online/streaming variant introduced in
//                               Phase 4; every row is fully materialized here so
//                               there is nothing to stream across).
//   3. weighted_sum_kernel:    out[b,h,i,d] = sum_j scores[b,h,i,j] * v[b,h,j,d].
//
// `scores_scratch` must point to B*H*S*S floats of device memory (caller-
// allocated; this is the "materializing scores/output intermediates" the
// V1 rung is defined by). `out` must point to B*H*S*D floats.
//
// Throws flashlite::CudaError (see error_check.cuh) if any launch or CUDA
// runtime call fails.
void launch_attention_naive(const float* q, const float* k, const float* v, float* out,
                             float* scores_scratch, int batch, int heads, int seq_len,
                             int head_dim, bool causal, cudaStream_t stream);

}  // namespace flashlite
