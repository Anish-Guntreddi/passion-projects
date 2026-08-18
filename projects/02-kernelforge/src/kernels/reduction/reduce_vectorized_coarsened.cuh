// Reduction V4 (vectorized + coarsened loads). ONE conceptual variable
// changes versus reduce_warp_shuffle.cuh: how many elements each thread
// loads, and how wide each individual load transaction is. The block-level
// combine (sequential addressing down to a warp, then warp-shuffle) is
// copy-identical to V3 -- see the loop below vs reduce_warp_shuffle.cu.
// (The spec's ladder -- 02-kernelforge-spec.md FR2 -- names "vectorized/
// coarsened loads" as a single ladder rung, not two separate ones; both
// changes serve the same one hypothesis below, so they are tested together
// here rather than split into two rungs.)
//
// HYPOTHESIS (written before this kernel existed):
// V1-V3 each launch ceil(n/block_size) threads, one element per thread,
// one 4-byte load per thread. Two effects should limit V3's throughput at
// large n: (a) 4-byte loads under-use the 128-byte memory-transaction
// width relative to what wider loads could pack per transaction, and (b)
// at large n, V3 launches a very large number of blocks, so the
// cross-block atomicAdd combine this whole ladder holds constant (see V1's
// header) contends more as n grows. V4 changes the load phase to: each
// thread grid-strides over 128-bit `float4` loads (4 floats/transaction
// instead of 1) and, because the grid is capped well below
// "one thread per float4" (kMaxBlocks in the .cu file), each thread also
// performs several loop iterations before entering the same block-reduce
// tail as V3 (thread coarsening) -- fewer, busier threads and fewer total
// blocks, hence fewer atomicAdd calls into the global accumulator. Expect
// a further speedup over V3 that grows with n (more opportunity for wider
// transactions to matter, and a proportionally larger cut in the number of
// blocks/atomics), with diminishing or possibly negative returns at very
// small n (coarsening a workload smaller than the grid cap just moves the
// same work onto fewer threads without adding new parallelism).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "common/occupancy.cuh"

namespace kernelforge::kernels {

// `zero_output` (default true): see reduce_naive_interleaved.cuh -- pass
// false only when the caller already zeroed `*d_out` on the same stream.
void reduce_vectorized_coarsened_launch(const float* d_in, float* d_out, std::size_t n,
                                         int block_size, cudaStream_t stream, dim3& out_grid,
                                         dim3& out_block, bool zero_output = true);

float reduce_vectorized_coarsened_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                            int block_size = 256);

// Phase 6: theoretical occupancy at `block_size` -- see common/occupancy.cuh.
kernelforge::OccupancyReport reduce_vectorized_coarsened_query_occupancy(int block_size = 256);

} // namespace kernelforge::kernels
