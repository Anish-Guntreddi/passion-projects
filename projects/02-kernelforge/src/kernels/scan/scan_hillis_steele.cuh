// Scan strategy A: Hillis-Steele (naive, step-efficient, double-buffered).
//
// HYPOTHESIS (written before scan_blelloch.cuh's competing strategy was
// implemented -- see src/kernels/scan/README.md for the side-by-side
// writeup and the measured comparison):
// Every one of the log2(block_size) passes keeps ALL block_size threads
// active (each either adds or passes its value through), so total work is
// O(block_size * log2(block_size)) per block -- more raw adds than a
// work-efficient algorithm needs, but every pass is trivially simple
// (uniform per-thread work, no address-pattern branching beyond a single
// `tid >= offset` compare, which is warp-uniform after the first pass)
// and needs only `log2(block_size)` synchronization points. Expect this
// to be competitive with, and possibly faster than, scan_blelloch.cuh at
// this GPU's block sizes (<=1024, so <=10 passes) precisely because GPUs
// are wide enough to absorb the redundant work in parallel, while
// Blelloch's two-phase tree pays a similar number of synchronization
// points for a more complex, harder-to-schedule access pattern. If
// measurements disagree, that is reported as-is (see docs/optimization-
// method.md step 6) rather than adjusted to match this prediction.
//
// Both strategies build on the shared multi-block orchestration in
// scan_common.cuh/.cu (which this file's launcher matches the
// BlockScanPassFn signature of) -- see that header for the 3-pass
// "scan, scan-of-sums, add-offsets" structure that gives this an
// arbitrary-n scan, not just a single-block one.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "kernels/scan/scan_common.cuh"

namespace kernelforge::kernels {

// Matches BlockScanPassFn's signature so it can be passed directly to
// scan_multilevel_launch. Computes an inclusive Hillis-Steele scan of each
// block_size-sized chunk of d_in into d_out, and each chunk's total into
// d_block_sums[blockIdx.x].
void scan_hillis_steele_block_pass_launch(const float* d_in, float* d_out, float* d_block_sums,
                                           std::size_t n, int block_size, cudaStream_t stream);

// Full multi-block inclusive scan using the Hillis-Steele per-block
// algorithm. Device-pointer launch: does not allocate (uses `scratch`,
// pre-sized by scan_alloc_scratch(block_size)); caller checks
// cudaGetLastError() and times as usual.
void scan_hillis_steele_launch(const float* d_in, float* d_out, const ScanScratch& scratch,
                                std::size_t n, int block_size, cudaStream_t stream,
                                dim3& out_grid, dim3& out_block);

// Host convenience wrapper: allocates d_in/d_out/scratch, copies h_in in,
// launches, synchronizes, copies h_out back, frees. Returns kernel-only
// elapsed ms. n == 0 is legal (empty scan) and short-circuits.
float scan_hillis_steele_run_host(const float* h_in, float* h_out, std::size_t n,
                                   int block_size = 256);

} // namespace kernelforge::kernels
