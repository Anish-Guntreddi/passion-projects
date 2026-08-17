// Scan strategy B: Blelloch (work-efficient, up-sweep/down-sweep).
//
// HYPOTHESIS (written before benchmarking against scan_hillis_steele.cuh
// -- see src/kernels/scan/README.md for the side-by-side writeup and the
// measured comparison):
// Hillis-Steele does O(block_size * log2(block_size)) total adds per
// block (every thread active every pass). Blelloch's up-sweep (reduce)
// phase followed by a down-sweep (distribute) phase does only
// O(block_size) total adds -- work-efficient -- at the cost of MORE
// distinct phases (2 * log2(block_size) instead of log2(block_size)) and
// an addressing pattern where the number of ACTIVE threads per step
// halves (up-sweep) or doubles (down-sweep) instead of staying at
// block_size the whole time, so later/earlier steps leave most of a warp
// idle. Expect Blelloch to move strictly less data through shared memory
// per block, but Hillis-Steele's simpler, always-fully-active steps may
// still win or tie at this GPU's supported block sizes (<=1024, so
// <=10 steps either way) because the synchronization-barrier count, not
// raw arithmetic, likely dominates at this scale -- work-efficiency
// matters more as block_size grows than it does at 1024. If measurements
// disagree, that is reported as-is (docs/optimization-method.md step 6).
//
// Standard, widely-taught two-phase algorithm (see e.g. any GPU-computing
// course's "work-efficient scan" lecture); written here from that
// well-known recurrence, not copied from any specific external codebase
// (hard constraint 2). This kernel internally computes an EXCLUSIVE scan
// (Blelloch's natural output) and adds the original per-element value
// before writing, so its *output* is INCLUSIVE -- identical output
// semantics to scan_hillis_steele.cuh, so both strategies share one CPU
// reference and one correctness test (tests/test_scan.cpp).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "kernels/scan/scan_common.cuh"

namespace kernelforge::kernels {

// Matches BlockScanPassFn's signature. Computes an inclusive Blelloch scan
// of each block_size-sized chunk of d_in into d_out, and each chunk's
// total into d_block_sums[blockIdx.x].
void scan_blelloch_block_pass_launch(const float* d_in, float* d_out, float* d_block_sums,
                                      std::size_t n, int block_size, cudaStream_t stream);

void scan_blelloch_launch(const float* d_in, float* d_out, const ScanScratch& scratch,
                           std::size_t n, int block_size, cudaStream_t stream, dim3& out_grid,
                           dim3& out_block);

float scan_blelloch_run_host(const float* h_in, float* h_out, std::size_t n,
                              int block_size = 256);

} // namespace kernelforge::kernels
