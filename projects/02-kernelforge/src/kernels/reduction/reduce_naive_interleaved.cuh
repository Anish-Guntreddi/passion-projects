// Reduction V1 (naive, atomic combine, interleaved addressing).
//
// NOTE: V0 (reduce_naive_global.cuh) was added after this file, as an even
// more naive pure-global-memory predecessor with no shared memory at all --
// see its header and src/kernels/reduction/README.md's "Ladder history"
// note for why. This file remains the first rung to introduce shared
// memory and is still the baseline every V2+ shared-memory-tree rung is
// measured against; only the "first GPU kernel in the ladder" framing below
// is now specifically "first shared-memory kernel."
//
// HYPOTHESIS (written before V2/V3/V4 exist -- see reduce_sequential_
// addressing.cuh, reduce_warp_shuffle.cuh, reduce_vectorized_coarsened.cuh
// for each successor's own hypothesis, and src/kernels/reduction/README.md
// for the full ladder in one place):
//
// This is the deliberately-naive baseline every later rung is measured
// against. It combines two textbook-naive choices in one kernel:
//   1. INTERLEAVED addressing in the shared-memory tree reduction
//      (`if (tid % (2*stride) == 0)`): only every 2*stride-th thread is
//      active per step. Within a warp (32 consecutive threadIdx.x values),
//      active and inactive threads interleave, so the warp scheduler must
//      serialize the "add" path and the "do nothing" path (warp
//      divergence) instead of every lane doing the same thing. This is the
//      primary mechanism this rung is naive about, and the one V2
//      (reduce_sequential_addressing.cuh) removes.
//        NOTE on shared-memory bank conflicts: an earlier version of this
//        comment also claimed this addressing pattern causes bank
//        conflicts on the early (small-stride) steps. Worked out by hand
//        for this GPU's 32 banks / 4-byte FP32 words: at stride s (a power
//        of two, s in {1,2,4,8,16}), the 16/s active lanes within one warp
//        touch addresses spaced 2s apart, and since 2s divides 32 evenly,
//        those addresses land on 16/s *distinct* banks with no repeats --
//        i.e. no bank conflict actually occurs here for either the
//        `sdata[tid]` or `sdata[tid+stride]` access at any of these
//        strides. That claim was dropped rather than left unverified (see
//        the finding this comment resolves); V1's slowdown vs V2 is
//        attributed to warp divergence alone. Re-verify with Nsight
//        Compute (`smsp__sass_average_data_bank_conflicts_pipe_lsu` or
//        similar) in Phase 6 before ever re-introducing a bank-conflict
//        claim here.
//   2. A single GLOBAL atomicAdd from each block's thread 0 to combine
//      block partial sums into the final scalar (avoids a second kernel
//      launch / multi-pass reduction). With grid size = ceil(n/block_size),
//      this is `ceil(n/block_size)` atomics all contending on the SAME
//      address -- a small-scale preview of the Phase 3 histogram/atomics
//      contention story, held constant across V1-V4 so it never becomes a
//      confounding second variable in this ladder (see
//      src/kernels/reduction/README.md).
//
// Expect this to be the slowest SHARED-MEMORY rung at every size (V0's
// pure-global-atomics baseline is expected to be slower still, per its own
// header), with the gap to V2 (which changes ONLY the addressing pattern)
// widening as n grows: more tree-reduction steps pay the warp-divergence
// tax before the surviving-thread count drops below one warp.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Device-pointer launch. `d_out` must point to one float. By default
// (`zero_output = true`) this function zeroes it (cudaMemsetAsync) before
// accumulating into it, so callers do not need to pre-zero it themselves.
// Pass `zero_output = false` only when the caller has already reset
// `*d_out` to 0 on the SAME stream (e.g. a benchmark harness that zeroes
// it before starting its timer specifically so the zero-fill doesn't get
// counted as kernel time -- see apps/bench_reduction_main.cpp). Does not
// synchronize; caller checks cudaGetLastError() (KF_CUDA_CHECK_LAST_ERROR())
// and times via common/gpu_timer.cuh as usual. block_size must be a power
// of two in [32, 1024] (common/launch_validate.hpp::validate_block_size_pow2_1d).
void reduce_naive_interleaved_launch(const float* d_in, float* d_out, std::size_t n,
                                      int block_size, cudaStream_t stream, dim3& out_grid,
                                      dim3& out_block, bool zero_output = true);

// Host convenience wrapper: allocates d_in/d_out, copies h_in in, launches,
// synchronizes, copies the scalar result into *h_sum_out, frees. Returns
// kernel-only elapsed ms (H2D/D2H copy excluded, FR4). n == 0 is legal
// (sum of the empty set is 0) and short-circuits without touching the GPU.
float reduce_naive_interleaved_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                         int block_size = 256);

} // namespace kernelforge::kernels
