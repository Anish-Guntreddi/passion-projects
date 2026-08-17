// Shared bin-index mapping used by every histogram kernel variant
// (Phase 3). This is a trivial, non-algorithmic utility (not part of any
// rung's distinguishing "atomics contention" mechanism), so it is shared
// -- the same rationale as kernels/scan/scan_common.cuh's glue kernels.
// `kernelforge::reference::histogram` (src/common/reference_ops.cpp)
// independently re-implements this exact formula (it cannot include this
// header -- src/common/ does not depend on src/kernels/, see that
// function's doc comment) and MUST stay in sync with it by hand for
// correctness tests to be meaningful.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Default max shared memory per block on this repo's one target GPU
// (RTX 4090 / sm_89, ADR 0001) without opting into the larger
// carveout via cudaFuncAttributePreferredSharedMemoryCarveout --
// verified by docs/gpu-model.md's `device-info` output ("Shared mem/block
// : 49152 bytes"). A fixed, disclosed hardware constant for this
// single-GPU repo, not a portability abstraction.
constexpr std::size_t kMaxSharedMemPerBlockBytes = 49152;

// Throws std::invalid_argument if a per-block private histogram of
// num_bins ints would not fit in shared memory. Called by every
// privatized histogram launcher before it requests that much dynamic
// shared memory (hard constraint 8: fail loudly instead of letting the
// CUDA runtime reject the launch after the fact).
inline void validate_num_bins_fits_shared_mem(int num_bins, const char* launcher_name) {
  const std::size_t needed = static_cast<std::size_t>(num_bins) * sizeof(int);
  if (num_bins <= 0 || needed > kMaxSharedMemPerBlockBytes) {
    throw std::invalid_argument(
        std::string(launcher_name) + ": num_bins (" + std::to_string(num_bins) +
        ") must be > 0 and its private histogram (" + std::to_string(needed) +
        " bytes) must fit within kMaxSharedMemPerBlockBytes (" +
        std::to_string(kMaxSharedMemPerBlockBytes) + " bytes).");
  }
}

// Every histogram kernel variant in this repo accumulates per-bin counts
// into a 32-bit `int` via atomicAdd (see e.g. histogram_global_atomic.cu).
// In the worst case (all n input elements landing in the same bin -- e.g.
// num_bins == 1, or an extremely skewed distribution) a single bin's
// counter receives up to `n` increments. If n exceeds INT_MAX, that
// counter can silently wrap into a negative/garbage value (signed integer
// overflow) with no runtime error. Guard against launching with an n that
// makes this possible at all, regardless of num_bins or input distribution
// (hard constraint 8: fail loudly on an unsupported case instead of
// letting it silently corrupt a result). This repo's kernels intentionally
// keep `int` counters (not widened to a 64-bit type) since every size this
// repo actually benchmarks (`benchmarks/configs/histogram_*.json`, largest
// n = 33,554,432) is far below this limit; widening the counter type is
// explicit future work if a use case ever needs n > INT_MAX.
inline void validate_n_fits_int_bin_counts(std::size_t n, const char* launcher_name) {
  constexpr std::size_t kMaxN = static_cast<std::size_t>((std::numeric_limits<int>::max)());
  if (n > kMaxN) {
    throw std::invalid_argument(
        std::string(launcher_name) + ": n (" + std::to_string(n) +
        ") exceeds " + std::to_string(kMaxN) +
        " (INT_MAX) -- a single bin's 32-bit atomicAdd counter could "
        "overflow if enough elements land in it (worst case: all of "
        "them, e.g. num_bins == 1). Reduce n or widen the histogram "
        "counter type (see this function's doc comment) before running "
        "this size.");
  }
}

// bin = clamp(floor(x * num_bins), 0, num_bins - 1). __host__ __device__
// so it compiles identically whether called from a kernel or (for
// completeness/documentation, though the CPU reference has its own
// independent copy) from host code.
__host__ __device__ inline int histogram_bin_of(float x, int num_bins) {
  int bin = static_cast<int>(x * static_cast<float>(num_bins));
  if (bin < 0) bin = 0;
  if (bin >= num_bins) bin = num_bins - 1;
  return bin;
}

} // namespace kernelforge::kernels
