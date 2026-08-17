// Histogram V1 (global-atomic baseline). Phase 3's contention baseline:
// every thread's atomicAdd targets the global histogram array directly,
// with no privatization of any kind.
//
// HYPOTHESIS (written before histogram_privatized.cuh existed -- see that
// file's own hypothesis and src/kernels/histogram/README.md for the full
// lab writeup and the measured low- vs high-contention comparison):
// Every one of the n input elements causes exactly one
// `atomicAdd(&hist[bin], 1)` directly against global memory. Under a LOW-
// contention (uniform) input distribution, the n atomics spread roughly
// evenly across num_bins distinct addresses, so two threads within the
// same warp rarely target the same address at the same time, and this
// kernel should be close to memory-bandwidth-bound (dominated by reading
// `in`, not by atomic serialization). Under a HIGH-contention (skewed)
// distribution (kernelforge::ContentionProfile::kSkewed --
// src/common/rng.hpp), the large majority of atomics target the SAME
// handful of addresses -- the hardware must serialize every colliding
// atomicAdd to that address, so this kernel's time should scale with
// contention, not just n: expect a measurable slowdown moving from the
// uniform to the skewed input at the SAME n, num_bins, block/grid
// configuration (the input distribution is the only variable in that
// comparison; see src/kernels/histogram/README.md).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Device-pointer launch. `d_hist` must point to num_bins ints. By default
// (`zero_output = true`) this function zeroes it (cudaMemsetAsync) before
// accumulating, so callers do not need to pre-zero it. Pass
// `zero_output = false` only when the caller already zeroed `d_hist` on
// the SAME stream (e.g. a benchmark harness that zeroes it before starting
// its timer specifically so the zero-fill doesn't get counted as kernel
// time -- see apps/bench_histogram_main.cpp). Does not synchronize.
void histogram_global_atomic_launch(const float* d_in, int* d_hist, std::size_t n, int num_bins,
                                     int block_size, cudaStream_t stream, dim3& out_grid,
                                     dim3& out_block, bool zero_output = true);

// Host convenience wrapper: allocates d_in/d_hist, copies h_in in,
// launches, synchronizes, copies h_hist_out back, frees. Returns
// kernel-only elapsed ms. n == 0 is legal (every bin is 0) and
// short-circuits without touching the GPU.
float histogram_global_atomic_run_host(const float* h_in, std::size_t n, int* h_hist_out,
                                        int num_bins, int block_size = 256);

} // namespace kernelforge::kernels
