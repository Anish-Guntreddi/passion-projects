// Vector add: out[i] = x[i] + y[i]. The Phase 0 "hello world" kernel used
// to validate the whole harness (CMake+CUDA build, error macros, device
// query, timers, reference comparison, benchmark schema) end to end before
// anything more interesting is built on top of it.
//
// Common conceptual interface (per spec §Tech-Stack "Repository structure"):
// input allocation/prep -> launch wrapper -> sync/error check -> output
// validation -> timing metadata. Every kernel family in this repo follows
// this same two-layer shape:
//   - `*_launch`: takes already-allocated device pointers, launches the
//     kernel, does not sync or copy anything. Used by benchmark code that
//     wants to time only the kernel across many repetitions without
//     alloc/copy overhead polluting the measurement.
//   - `*_run_host`: takes host pointers, does alloc + H2D copy + launch +
//     sync + D2H copy + free, and returns kernel-only elapsed ms (measured
//     with CUDA events around the launch only, not the copies). Used by
//     correctness tests, where convenience matters more than avoiding
//     alloc overhead.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Device-pointer launch: d_x, d_y, d_out must already be allocated with
// space for `n` floats. Does not synchronize; caller is responsible for
// synchronization/timing (see common/gpu_timer.cuh) and for checking
// cudaGetLastError() after this returns (KF_CUDA_CHECK_LAST_ERROR()).
void vector_add_launch(const float* d_x, const float* d_y, float* d_out,
                        std::size_t n, int block_size, cudaStream_t stream,
                        dim3& out_grid, dim3& out_block);

// Host-pointer convenience wrapper for correctness tests: allocates device
// buffers, copies h_x/h_y in, launches, synchronizes, copies h_out back,
// frees. Returns the kernel-only elapsed time in milliseconds (H2D/D2H
// copy time is not included, per FR4: never mix kernel and transfer time
// unless explicitly measuring end-to-end).
float vector_add_run_host(const float* h_x, const float* h_y, float* h_out,
                           std::size_t n, int block_size = 256);

} // namespace kernelforge::kernels
