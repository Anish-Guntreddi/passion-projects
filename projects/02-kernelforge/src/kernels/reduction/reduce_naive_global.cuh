// Reduction V0-GPU (naive, pure global-memory, atomic combine).
//
// This is the literal FR2 "V1 naive global-memory" rung (02-kernelforge-
// spec.md Part 1.3): the CPU reference (`kernelforge::reference::
// reduce_sum`) is the ladder's V0, this file is the first GPU rung, and
// `reduce_naive_interleaved.cuh` (kept under its original name/identifier
// for benchmark-data continuity -- see this family's README.md "Ladder
// history" note) is the FIRST rung to introduce shared memory. Before this
// file existed, that shared-memory rung was the earliest GPU variant, which
// collapsed FR2's distinct "naive global-memory" and "shared-memory" stages
// into one; this file restores the missing, separately-preserved
// global-memory-only baseline (hard constraint 5: keep reference and
// optimized variants side-by-side for learning).
//
// HYPOTHESIS (written before this kernel existed):
// Every thread reads exactly one element from global memory and combines it
// into the running total with a SINGLE GLOBAL atomicAdd -- no shared
// memory, no tree reduction, no per-block staging of any kind. With
// `ceil(n/block_size)` blocks all issuing up to `block_size` threads' worth
// of atomicAdd on the SAME address, every one of those n additions
// serializes against every other one (the atomic unit itself is the
// bottleneck, not memory bandwidth). Expect this to be dramatically slower
// than every later rung -- including V1 (`reduce_naive_interleaved`), whose
// per-block shared-memory tree already cuts the number of global atomics
// from n down to `ceil(n/block_size)` -- and for the gap to widen sharply
// with n, since contention (not data volume) dominates this kernel's cost.
// This rung exists to make that "atomics-are-the-bottleneck" mechanism
// directly measurable in isolation, not to be a realistic kernel anyone
// would ship.
//
// Unlike every later rung, this kernel touches no shared memory and has no
// tree-addressing step, so it does not need `block_size` to be a power of
// two -- only the generic `validate_block_size_1d` bound applies (see
// reduce_naive_global.cu). Benchmarks still run it at the same block_size
// as the rest of the ladder (see benchmarks/configs/reduction.json) so
// results stay directly comparable.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// `zero_output` (default true): zero `*d_out` via cudaMemsetAsync before
// accumulating. Pass false only when the caller already zeroed it on the
// same stream (see reduce_naive_interleaved.cuh for the full rationale --
// used by apps/bench_reduction_main.cpp to keep that zero-fill out of the
// timed interval).
void reduce_naive_global_launch(const float* d_in, float* d_out, std::size_t n, int block_size,
                                 cudaStream_t stream, dim3& out_grid, dim3& out_block,
                                 bool zero_output = true);

// Host convenience wrapper: allocates d_in/d_out, copies h_in in, launches,
// synchronizes, copies the scalar result into *h_sum_out, frees. Returns
// kernel-only elapsed ms (H2D/D2H copy excluded, FR4). n == 0 is legal
// (sum of the empty set is 0) and short-circuits without touching the GPU.
float reduce_naive_global_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                    int block_size = 256);

} // namespace kernelforge::kernels
