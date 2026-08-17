// V2: tiled custom attention kernels (spec ladder SS1.3: "Tiled computation
// using shared/on-chip memory"; roadmap Phase 3: "Q/K/V tiled with
// shared/on-chip memory; conceptually simple softmax retained").
//
// Mathematical invariant (docs/attention-math.md, unchanged from V1):
//     S = (Q K^T) / sqrt(d);  P = softmax(S);  O = P V
// checked against flashlite.reference.attention (V0) elementwise within
// tolerance, exactly like V1 (tests/correctness/test_tiled_attention.py).
//
// Tensor shape/layout contract: identical to V1 (ADR 0002/0003/0004,
// enforced by bindings.cpp via the shared cuda_common/shape_validate.hpp
// before any kernel launches) -- q, k, v are [B, H, S, D] contiguous
// float32 CUDA tensors, all three the same shape, `causal` masks key
// position j > query position i with -inf before softmax. Any positive
// S/D, including S values that are not a multiple of kAttnTileDim (see
// the boundary-check comments in attention_tiled.cu).
//
// ONE variable changes versus V1 (docs/attention-math.md SS5's ladder
// description, and the KernelForge-style "hypothesis before measurement"
// convention from src/kernels/gemm/gemm_tiled.cuh there):
//
// HYPOTHESIS (written in docs/io-analysis.md before this kernel existed,
// restated here next to the code it predicts): V1's compute_scores_kernel
// and weighted_sum_kernel each have every thread independently re-read its
// own D-length (resp. S-length) row/column straight from global memory,
// with zero reuse across the many threads that share the same Q row, K
// row, P row, or V column -- exactly the "materializing scores/output
// intermediates" the V1 rung is defined by (attention_naive.cuh), and the
// dominant term in the theoretical no-reuse byte count docs/io-analysis.md
// derives for it. This file replaces ONLY those two kernels' inner loop
// with the same shared-memory cooperative-tile-load pattern KernelForge's
// gemm_tiled_kernel uses for a plain GEMM (each kAttnTileDim x kAttnTileDim
// tile of Q/K/P/V is read from global memory exactly once per output tile
// it contributes to, staged through shared memory, and reused
// kAttnTileDim times by every thread in the block that needs it) --
// reducing the QK^T and PV global-memory read terms by a factor of
// kAttnTileDim, while leaving the row-softmax kernel (kernel 2) and the
// full [B, H, S, S] materialization between kernels completely UNCHANGED
// ("conceptually simple softmax retained" is the roadmap's literal
// wording: Phase 4 is what changes the softmax algorithm itself, not this
// phase). docs/io-analysis.md predicts the resulting arithmetic-intensity
// change quantitatively and is written before this kernel's benchmark
// numbers exist; benchmarks/methodology.md SS9 records what was actually
// measured against that prediction.
#pragma once

#include <cuda_runtime.h>

namespace flashlite {

// Tile dimension for both the thread block (kAttnTileDim x kAttnTileDim
// threads) and the shared-memory sub-tiles of Q/K/P/V (kAttnTileDim x
// kAttnTileDim floats each). 32 matches this GPU's warp size (coalescing
// granularity) and KernelForge's own gemm_tiled precedent
// (kGemmTileDim=32, ../02-kernelforge/src/kernels/gemm/gemm_tiled.cuh);
// 2 tiles x 32 x 32 x 4 bytes = 8192 bytes of shared memory per block for
// each of kernel 1 and kernel 3, well within this GPU's 49,152-byte
// default per-block budget (docs/gpu-model.md in ../02-kernelforge). This
// is a PROVISIONAL default, not yet the spec's D5 answer -- ADR 0007
// documents why 32 was chosen for Phase 3 and that Phase 6 resolves D5
// (shared-memory vs. register-pressure tile-size tradeoffs) empirically,
// which may change this constant.
constexpr int kAttnTileDim = 32;

// Launches the 3-kernel tiled attention pipeline on `stream`:
//   1. compute_scores_tiled_kernel: scores[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) * scale,
//                                    masked to -inf where causal && j > i.
//                                    Q and K tiles are staged through shared
//                                    memory and reused across the block
//                                    (the one variable this rung changes
//                                    for kernel 1; see this file's header).
//   2. softmax_rows_kernel:         scores[b,h,i,:] <- softmax(scores[b,h,i,:]) in place.
//                                    Byte-for-byte identical algorithm to V1's
//                                    softmax_rows_kernel (two-pass, block-per-row,
//                                    shared-memory max/sum reduction) --
//                                    intentionally duplicated, not shared,
//                                    per "variants live side-by-side" (spec
//                                    Part 2); this is NOT the variable Phase 3
//                                    changes.
//   3. weighted_sum_tiled_kernel:   out[b,h,i,d] = sum_j scores[b,h,i,j] * v[b,h,j,d].
//                                    P and V tiles staged through shared memory
//                                    and reused across the block (same tiling
//                                    idea as kernel 1, applied to a plain,
//                                    non-transposed GEMM shape).
//
// `scores_scratch` must point to B*H*S*S floats of device memory
// (caller-allocated, same materialization contract as V1 -- Phase 3 tiles
// the computation, it does not remove the materialized intermediate; that
// is Phase 5's job). `out` must point to B*H*S*D floats.
//
// Throws flashlite::CudaError (see cuda_common/error_check.cuh) if any
// launch or CUDA runtime call fails.
void launch_attention_tiled(const float* q, const float* k, const float* v, float* out,
                             float* scores_scratch, int batch, int heads, int seq_len, int head_dim,
                             bool causal, cudaStream_t stream);

}  // namespace flashlite
