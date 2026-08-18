// V4: fused, IO-aware attention (spec ladder SS1.3: "Fused tiled attention
// avoiding full score/probability materialization"; roadmap Phase 5: "Full
// attention-matrix materialization removed; output accumulated
// tile-by-tile"). This is the single-kernel, FlashAttention-STYLE
// formulation the project brief describes -- derived independently from
// docs/online-softmax.md's combine identity (spec hard constraint 1: never
// copy the production FlashAttention implementation; hard constraint 3:
// keep the optimized implementation readable), not ported from any
// existing library.
//
// Mathematical invariant (docs/attention-math.md, unchanged from
// V0-V3): S = (Q K^T) / sqrt(d);  P = softmax(S);  O = P V. Checked
// against flashlite.reference.attention (V0) elementwise within tolerance
// (tests/correctness/test_fused_attention.py).
//
// Tensor shape/layout contract: [B, H, S, D] contiguous float32 CUDA
// tensors, q/k/v sharing one shape (ADR 0002/0003/0004, enforced by
// bindings.cpp via cuda_common/shape_validate.hpp), PLUS one V4-specific
// addition: `head_dim <= kMaxHeadDimFused` (see below) -- every shape this
// repo's test matrix uses (32/64/128) satisfies this; a clear
// TORCH_CHECK failure documents the boundary otherwise ("unsupported
// shapes must fail clearly", spec SS1.7).
//
// ONE variable changes versus V3 (this repo's "one optimization variable
// per variant" convention) -- and it is the biggest single change in the
// whole ladder, because it is the point of the entire project:
//
// V3 (src/flashlite/cuda_online_softmax/) replaced V1/V2's two-pass
// softmax with a single-combined-pass ONLINE softmax, but that online
// update still normalized an ALREADY-materialized [B, H, S, S] row (kernel
// 1 writes the full score matrix to global memory; kernel 2 reads it back;
// kernel 3 reads the normalized probabilities back again). V4 applies the
// IDENTICAL per-element online update rule (docs/online-softmax.md SS3/SS5,
// the exact combine identity src/flashlite/online_softmax.py's
// `combine_online_stats`/`online_softmax_attention_row` already prove
// correct on CPU, independent of any kernel) directly to Q K^T scores AS
// THEY ARE COMPUTED, one K/V tile at a time, entirely within ONE kernel
// launch, with NO scores buffer at all: `attention_fused_forward`
// (bindings.cpp) allocates only `out` ([B, H, S, D]) -- there is no
// `torch::empty({B, H, S, S}, ...)` anywhere in this file's Python-facing
// entry point, unlike every earlier variant. This is the literal,
// verifiable meaning of "full attention-matrix materialization removed"
// (roadmap Phase 5) and "O(S*D) memory traffic per slice, matching the
// irreducible lower bound" (docs/attention-math.md SS5's forward
// reference, docs/io-analysis.md SS2's `AI_ideal` argument) --
// tests/correctness/test_fused_attention.py's peak-memory test measures
// this directly (torch.cuda.max_memory_allocated()), not just asserts it.
//
// Design (ADR 0011 records the full reasoning): one thread per query row,
// one thread block covers kFusedTileDim query rows (a "Q tile"), and each
// block iterates sequentially over K/V in kFusedTileDim-row tiles ("KV
// tiles" -- the literal read of "online softmax ACROSS K/V TILES", spec
// SS1.3's V3 description, realized here with no scores buffer at all
// rather than V3's row-chunking-of-an-already-materialized-row). Each
// tile's K/V rows are cooperatively staged through shared memory (same
// staging pattern V2/V3's kernel 1/3 already use); each thread then
// updates its own private (m, l, acc[D]) triple with the online recurrence
// for every key column in that tile (skipping columns masked by
// causality), before the block moves to the next tile. After the last
// tile, each thread normalizes its own row (acc / l) and writes it once.
#pragma once

#include <cuda_runtime.h>

namespace flashlite {

// Rows-per-block (and KV-rows-per-streamed-tile) dimension. 32 matches this
// GPU's warp size and V2/V3's kAttnTileDim/kOnlineSoftmaxAttnTileDim, for
// the identical reasons ADR 0007 already gives (warp-size-exact
// coalescing, consistency with this repo's sibling tiled kernels) --
// ADR 0011 records why V4 reuses this value rather than tuning a new one
// (deferred to Phase 6, same as ADR 0007 defers D5).
constexpr int kFusedTileDim = 32;

// Upper bound on head_dim this kernel supports: each thread holds a
// private Q row (`q_reg[kMaxHeadDimFused]`) and output accumulator
// (`acc[kMaxHeadDimFused]`) as fixed-size register/local arrays (CUDA
// device-code array sizes must be compile-time constants). 128 covers
// every head_dim this repo's test matrix uses (spec SS1.7: "head_dim
// 32/64/128"); ADR 0011 documents the register-pressure tradeoff this
// implies and why it is an accepted Phase 5 characteristic (Phase 6 tunes
// tile sizes/register pressure empirically, per spec D5), not a Phase 5
// defect.
constexpr int kMaxHeadDimFused = 128;

// Launches the SINGLE fused attention kernel on `stream`. `out` must point
// to B*H*S*D floats; there is intentionally no scores-buffer parameter at
// all (see this file's header). Requires `head_dim <= kMaxHeadDimFused`
// (checked by bindings.cpp before this is called, not re-checked here --
// see cuda_common/shape_validate.hpp's convention of validating once at the
// pybind11 boundary, kernels themselves assume valid input).
//
// Throws flashlite::CudaError (see cuda_common/error_check.cuh) if any
// launch or CUDA runtime call fails.
void launch_attention_fused(const float* q, const float* k, const float* v, float* out, int batch,
                             int heads, int seq_len, int head_dim, bool causal, cudaStream_t stream);

}  // namespace flashlite
