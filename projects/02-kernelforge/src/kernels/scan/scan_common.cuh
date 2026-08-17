// Shared multi-block scan orchestration, used identically by every scan
// strategy (scan_hillis_steele.cu, scan_blelloch.cu). This is
// infrastructure/plumbing shared on purpose (same rationale as
// apps/bench_cli.hpp being shared across every bench_*_main.cpp) -- it is
// NOT part of either strategy's distinguishing algorithm, so sharing it
// does not compromise the "keep every ladder rung side-by-side" goal
// (hard constraint 5): each strategy's actual scan algorithm (Hillis-
// Steele vs Blelloch) still lives entirely in its own .cu file.
//
// ## The 3-pass structure
// A single thread block can only scan up to `block_size` elements (its
// own shared-memory tile). To scan an arbitrary n, this uses the standard
// "scan, scan-of-sums, add-offsets" decomposition:
//   1. Each block computes an INCLUSIVE scan of its own chunk of up to
//      `block_size` elements into `d_out`, and separately records that
//      chunk's total into `d_block_sums[blockIdx.x]` (a per-block
//      `BlockScanPassFn`, provided by the strategy).
//   2. If there is more than one block, the (small, <= block_size)
//      `d_block_sums` array is itself scanned in a SINGLE block (by
//      calling the SAME BlockScanPassFn again, with its own block
//      dimension sized to fully cover num_blocks) to get each block's
//      inclusive prefix; subtracting that block's own sum converts it to
//      an EXCLUSIVE prefix -- the "carry-in" every element of that block
//      needs added.
//   3. A generic (strategy-independent) kernel broadcasts each block's
//      carry-in across every element of its chunk in `d_out`.
//
// ## Known, disclosed scope limit
// Step 2 requires num_blocks (= ceil(n / block_size)) to itself fit in one
// block, i.e. num_blocks <= block_size. This implementation therefore
// supports n up to block_size^2 elements (block_size = 1024 -> up to
// 1,048,576). Beyond that, a 3rd scan level would be required (recursing
// the same idea one level deeper); this is explicit future work, not
// silently included or worked around -- scan_multilevel_launch throws
// std::runtime_error rather than silently truncating or producing a wrong
// answer (hard constraint 8: fail loudly).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// Signature every per-block inclusive-scan kernel launcher must match.
// Computes an INCLUSIVE scan of `d_in[0..n)` into `d_out[0..n)`, one
// thread block per chunk of `block_size` elements (grid size is exactly
// ceil(n / block_size), computed internally). Elements at index >= n
// within the last (partial) block are treated as the additive identity
// (0). Writes each block's chunk total into `d_block_sums[blockIdx.x]`.
// `block_size` must be a power of two (validate_block_size_pow2_1d).
using BlockScanPassFn = void (*)(const float* d_in, float* d_out, float* d_block_sums,
                                  std::size_t n, int block_size, cudaStream_t stream);

// Pre-allocated scratch space for scan_multilevel_launch, sized once by
// the caller for a given block_size (so allocation never happens inside
// the timed region, matching every other kernel family's `_launch`
// convention). Every array holds `block_size` floats -- an upper bound on
// num_blocks for any n this implementation supports (see header comment).
struct ScanScratch {
  float* d_block_sums = nullptr;         // level-1 block totals
  float* d_block_sums_scanned = nullptr; // level-2 inclusive scan of the above
  float* d_block_offsets = nullptr;      // level-2 result converted to exclusive carry-ins
  float* d_level2_dummy_sums = nullptr;  // level-2 call's own (unused) block_sums output
};

// Allocates a ScanScratch whose arrays are each `block_size` floats
// (`d_level2_dummy_sums` is 1 float). Throws (via KF_CUDA_CHECK) on
// allocation failure.
ScanScratch scan_alloc_scratch(int block_size);

// Frees every buffer in `scratch` and resets its pointers to nullptr.
void scan_free_scratch(ScanScratch& scratch);

// Runs the full 3-pass multi-block inclusive scan described above.
// `d_in`/`d_out` must each hold n floats; `scratch` must have been sized
// for at least this `block_size` (scan_alloc_scratch(block_size)). Throws
// std::runtime_error if n exceeds this implementation's block_size^2
// ceiling (see header comment). n == 0 is legal and a no-op.
void scan_multilevel_launch(BlockScanPassFn block_scan_pass, const float* d_in, float* d_out,
                             const ScanScratch& scratch, std::size_t n, int block_size,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block);

} // namespace kernelforge::kernels
