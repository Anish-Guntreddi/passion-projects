// Shared launch-geometry validation for the 1D-block kernel launch wrappers
// (vector_add, saxpy, stride_copy). Hard constraint 8 ("fail loudly on
// unsupported hardware features") means an invalid block_size must throw
// before it reaches unsigned/dim3 arithmetic, not silently corrupt grid
// math or wait for the CUDA runtime to reject the launch after the fact.
#pragma once

#include <stdexcept>
#include <string>

namespace kernelforge {

// 1024 threads/block is the architectural ceiling on every CUDA compute
// capability this repo supports (ADR 0002's floor is 8.9/Ada; 1024 has been
// the max threads/block on every architecture since Fermi, cc 2.0 -- see
// docs/gpu-model.md). The vector/stride kernel families here launch 1D
// blocks only, so this single bound fully covers block_size for them.
constexpr int kMaxThreadsPerBlock1D = 1024;

// Throws std::invalid_argument if block_size is not in [1, 1024].
//
// Without this check, callers of vector_add_launch/saxpy_launch/
// stride_copy_launch could trigger:
//   - block_size == 0: `(n + block_size - 1) / block_size` divides by
//     zero in host-side grid-size arithmetic -- undefined behavior
//     (SIGFPE on x86) before any GPU code runs.
//   - block_size < 0: sign-converts to a huge `unsigned` when building
//     `dim3 block(static_cast<unsigned>(block_size))`, which the CUDA
//     runtime only rejects (cudaErrorInvalidConfiguration) after the
//     launch has already been queued.
//   - block_size > 1024: same late rejection by the CUDA runtime instead
//     of a clear, immediate, catchable error.
inline void validate_block_size_1d(int block_size, const char* launcher_name) {
  if (block_size < 1 || block_size > kMaxThreadsPerBlock1D) {
    throw std::invalid_argument(
        std::string(launcher_name) + ": block_size (" + std::to_string(block_size) +
        ") must satisfy 1 <= block_size <= " + std::to_string(kMaxThreadsPerBlock1D));
  }
}

} // namespace kernelforge
