// Histogram V3 (privatized + thread coarsening). ONE variable changes
// versus histogram_privatized.cuh: how many input elements each thread
// processes before the block-private histogram is merged into the global
// one. The privatization strategy itself (shared-memory private
// histogram, atomicAdd locally, merge once) is unchanged.
//
// HYPOTHESIS (written before this kernel existed):
// V2 launches ceil(n/block_size) blocks, each merging its private
// histogram with num_bins global atomics -- total global atomic traffic
// is O((n/block_size) * num_bins). If each thread instead processes
// `elems_per_thread` consecutive elements (coarsening) before that
// block's single merge, the number of blocks needed for the same n drops
// by that same factor, so total global merge traffic drops proportionally
// too: O((n/(block_size*elems_per_thread)) * num_bins). Local (shared-
// memory) atomic traffic is unchanged (still one atomicAdd per input
// element, just now spread over fewer, busier blocks). Expect a further
// improvement over V2, concentrated in the GLOBAL-merge-atomic cost --
// i.e. most visible at HIGH contention with a large num_blocks (small
// block_size or very large n), where V2's merge phase is a bigger
// fraction of total time; expect little-to-no change at LOW contention
// where V2 was already close to bandwidth-bound. See
// src/kernels/histogram/README.md for the measured numbers.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// `elems_per_thread` (>= 1) is the coarsening factor: each thread reads
// this many consecutive elements (a grid-stride-free contiguous chunk per
// block) before the block's single global merge.
// `zero_output` (default true): see histogram_global_atomic.cuh -- pass
// false only when the caller already zeroed `d_hist` on the same stream.
void histogram_privatized_coarsened_launch(const float* d_in, int* d_hist, std::size_t n,
                                            int num_bins, int block_size, int elems_per_thread,
                                            cudaStream_t stream, dim3& out_grid, dim3& out_block,
                                            bool zero_output = true);

float histogram_privatized_coarsened_run_host(const float* h_in, std::size_t n, int* h_hist_out,
                                               int num_bins, int block_size = 256,
                                               int elems_per_thread = 8);

} // namespace kernelforge::kernels
