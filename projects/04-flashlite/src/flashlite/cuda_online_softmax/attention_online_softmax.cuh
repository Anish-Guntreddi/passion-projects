// V3: online-softmax custom attention kernels (spec ladder SS1.3:
// "Numerically stable online softmax across K/V tiles"; roadmap Phase 4:
// "Running max + normalization accumulator implemented/tested
// independently, then integrated").
//
// Mathematical invariant (docs/attention-math.md, unchanged from V1/V2):
//     S = (Q K^T) / sqrt(d);  P = softmax(S);  O = P V
// checked against flashlite.reference.attention (V0) elementwise within
// tolerance (tests/correctness/test_online_softmax_attention.py).
//
// Tensor shape/layout contract: identical to V1/V2 (ADR 0002/0003/0004,
// enforced by bindings.cpp via cuda_common/shape_validate.hpp before any
// kernel launches).
//
// ONE variable changes versus V2 (this repo's "one optimization variable
// per variant" convention, matching attention_tiled.cuh's own header):
//
// V2's softmax_rows_kernel (attention_tiled.cu, byte-for-byte V1's) is a
// TWO-PASS, block-per-row reduction: one full pass over the row to find its
// max, a SECOND full pass to sum exp(x - max), then a third pass to
// normalize and write -- three full reads of every row element from global
// memory, because that algorithm cannot safely start summing until the max
// is fully known (docs/attention-math.md SS2). This file replaces ONLY
// that kernel with online_softmax_rows_kernel: each thread folds its own
// strided subset of the row into a running (m, l) pair using the online
// update rule docs/online-softmax.md derives (no need to know the max in
// advance -- the running max self-corrects as bigger values are found),
// then a block-level tree reduction COMBINES every thread's partial (m, l)
// into the row's (m, l) with the identical combine identity
// (docs/online-softmax.md SS3, src/flashlite/online_softmax.py's
// `combine_online_stats`, tested independently on CPU in
// tests/math/test_online_softmax.py BEFORE this kernel was written) --
// ONE combined max+sum pass, plus the same unavoidable final
// normalize-and-write pass every variant needs: TWO passes over the row
// instead of three. Kernel 1 (compute_scores_tiled_kernel) and kernel 3
// (weighted_sum_tiled_kernel) are UNCHANGED from V2 (duplicated here, not
// shared, per "variants live side-by-side," spec Part 2) -- the full
// [B, H, S, S] score/probability matrix is STILL fully materialized
// between kernels; removing that materialization entirely is Phase 5's
// (V4's) job, not this one's (docs/online-softmax.md SS10 states this
// boundary explicitly).
#pragma once

#include <cuda_runtime.h>

namespace flashlite {

// Same tile dimension as V2 (attention_tiled.cuh's kAttnTileDim), reused
// unchanged here for kernel 1/3's cooperative Q/K and P/V shared-memory
// staging -- ADR 0007 already covers why 32, and this variant does not
// change that decision (it changes kernel 2's algorithm, not kernel 1/3's
// tiling), so no new tile-size ADR is needed for kernels 1/3.
constexpr int kOnlineSoftmaxAttnTileDim = 32;

// Block size for kernel 2 (online_softmax_rows_kernel), one block per
// (b, h, i) row -- matches V1/V2's kSoftmaxBlockSize=256 exactly (same
// reduction-halving power-of-two requirement, same "unrelated to the QK^T
// tile dimension" reasoning attention_tiled.cu's kernel 2 comment already
// gives). ADR 0009 records why the online algorithm reuses this constant
// rather than introducing a new one.
constexpr int kOnlineSoftmaxBlockSize = 256;

// Launches the 3-kernel online-softmax attention pipeline on `stream`:
//   1. compute_scores_tiled_kernel: identical to V2's (attention_tiled.cu).
//   2. online_softmax_rows_kernel:  scores[b,h,i,:] <- softmax(scores[b,h,i,:])
//                                    in place, via the online running-max/
//                                    running-sum combine rule (this file's
//                                    header, docs/online-softmax.md) instead
//                                    of V1/V2's two-pass algorithm.
//   3. weighted_sum_tiled_kernel:   identical to V2's (attention_tiled.cu).
//
// `scores_scratch` must point to B*H*S*S floats of device memory
// (caller-allocated -- V3 still materializes the full score/probability
// matrix, per this file's header). `out` must point to B*H*S*D floats.
//
// Throws flashlite::CudaError (see cuda_common/error_check.cuh) if any
// launch or CUDA runtime call fails.
void launch_attention_online_softmax(const float* q, const float* k, const float* v, float* out,
                                      float* scores_scratch, int batch, int heads, int seq_len,
                                      int head_dim, bool causal, cudaStream_t stream);

}  // namespace flashlite
