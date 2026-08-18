// Reduction V2 (sequential addressing). Exactly ONE variable changes
// versus reduce_naive_interleaved.cuh: the shared-memory tree reduction's
// active-thread addressing pattern. Block size, grid size, the load phase,
// and the cross-block atomicAdd combine are all identical to V1.
//
// HYPOTHESIS (written before this kernel existed):
// Replacing `if (tid % (2*stride) == 0)` with `if (tid < stride)` makes
// the active-thread predicate a function of tid alone, split at a single
// threshold. Every thread in a given warp (32 consecutive threadIdx.x
// values) is on the SAME side of that threshold until stride drops below
// 32, so warp divergence during the tree reduction is eliminated for all
// but the last few (stride < 32) steps -- versus V1, where it existed at
// every step. This is the mechanism this rung's speedup is attributed to.
// (`sdata[tid]` and `sdata[tid+stride]` are also now read by
// spatially-adjacent threads at a fixed stride rather than an
// index-mod-dependent one; V1's header explains why, on this GPU's 32
// FP32-word banks, V1's early-stride accesses turn out not to actually
// bank-conflict either way, so that is not part of this rung's win.)
// Expect a clear speedup over V1 at every size, growing with
// block occupancy (bigger blocks -> more tree levels that used to diverge).
// This does NOT touch the cross-block atomicAdd combine, so any remaining
// gap to a hypothetical atomic-free reduction is out of scope for this
// rung (see V1's header for why atomics are held constant across the
// whole ladder).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "common/occupancy.cuh"

namespace kernelforge::kernels {

// `zero_output` (default true): zero `*d_out` via cudaMemsetAsync before
// accumulating. Pass false only when the caller already zeroed it on the
// same stream (see reduce_naive_interleaved.cuh for the full rationale --
// used by apps/bench_reduction_main.cpp to keep that zero-fill out of the
// timed interval).
void reduce_sequential_addressing_launch(const float* d_in, float* d_out, std::size_t n,
                                          int block_size, cudaStream_t stream, dim3& out_grid,
                                          dim3& out_block, bool zero_output = true);

float reduce_sequential_addressing_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                             int block_size = 256);

// Phase 6: theoretical occupancy at `block_size`, with this kernel's actual
// `block_size * sizeof(float)` dynamic shared-memory allocation -- see
// common/occupancy.cuh.
kernelforge::OccupancyReport reduce_sequential_addressing_query_occupancy(int block_size = 256);

} // namespace kernelforge::kernels
