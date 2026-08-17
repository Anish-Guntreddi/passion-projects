// Histogram V2 (shared-memory privatization). ONE variable changes versus
// histogram_global_atomic.cuh: where the atomics land during
// accumulation. Load phase, block/grid sizing, and bin formula are
// unchanged.
//
// HYPOTHESIS (written before this kernel existed):
// V1 sends every one of the n atomicAdds straight to global memory. This
// kernel instead gives each thread BLOCK a private copy of the histogram
// in shared memory: threads atomicAdd into that private copy (shared-
// memory atomics have far lower latency and much higher aggregate
// throughput per SM than global-memory atomics on this architecture),
// then, once, each block merges its private histogram into the global one
// with exactly num_bins atomicAdds (skipping bins that stayed at 0).
// Global atomic traffic drops from `n` total atomics (V1) to at most
// `num_blocks * num_bins` (this kernel) -- a large reduction whenever
// n >> num_blocks * num_bins, which is true for every size this repo
// benchmarks. Expect this to help BOTH contention profiles versus V1, but
// by a much larger margin under HIGH contention (skewed input): V1's
// worst case is serialized on a handful of GLOBAL addresses across the
// entire launch, while this kernel's worst case is serialized on a
// handful of SHARED addresses within one block at a time (cheaper
// per-collision, and the collisions across different blocks' private
// copies no longer contend with each other at all). Under LOW contention
// (uniform input), V1 was already close to bandwidth-bound, so the
// improvement here should be smaller. See
// src/kernels/histogram/README.md for the measured numbers.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// `zero_output` (default true): see histogram_global_atomic.cuh -- pass
// false only when the caller already zeroed `d_hist` on the same stream.
void histogram_privatized_launch(const float* d_in, int* d_hist, std::size_t n, int num_bins,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block, bool zero_output = true);

float histogram_privatized_run_host(const float* h_in, std::size_t n, int* h_hist_out,
                                     int num_bins, int block_size = 256);

} // namespace kernelforge::kernels
