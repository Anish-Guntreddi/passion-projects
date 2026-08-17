#include "kernels/scan/scan_common.cuh"

#include <stdexcept>
#include <string>

#include "common/error_check.cuh"
#include "common/launch_validate.hpp"

namespace kernelforge::kernels {

namespace {

// Strategy-independent glue kernels (see scan_common.cuh's "3-pass
// structure" comment, steps 2b/3). Neither depends on whether the
// per-block scan was Hillis-Steele or Blelloch.

// offsets[i] = inclusive_block_sums[i] - block_sums[i]  (turns an
// inclusive scan of block totals into each block's EXCLUSIVE carry-in).
__global__ void scan_compute_block_offsets_kernel(const float* __restrict__ inclusive_block_sums,
                                                    const float* __restrict__ block_sums,
                                                    float* __restrict__ offsets,
                                                    std::size_t num_blocks) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_blocks) {
    offsets[i] = inclusive_block_sums[i] - block_sums[i];
  }
}

// Broadcasts each block's carry-in across every element of its chunk.
__global__ void scan_add_block_offsets_kernel(float* __restrict__ out,
                                                const float* __restrict__ offsets, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] += offsets[blockIdx.x];
  }
}

int next_pow2_at_least(std::size_t x, int floor_value) {
  int p = floor_value;
  while (static_cast<std::size_t>(p) < x) p <<= 1;
  return p;
}

} // namespace

ScanScratch scan_alloc_scratch(int block_size) {
  // Validated here (not just inside scan_multilevel_launch) because this
  // function runs FIRST in the run_host convenience path and computes an
  // allocation size directly from block_size: an unvalidated negative
  // block_size wraps to a huge std::size_t, turning a bad-input error into
  // a confusing raw CUDA out-of-memory failure instead of a clear
  // std::invalid_argument (hard constraint 8: fail loudly, at the actual
  // point the bad value would otherwise cause harm).
  validate_block_size_pow2_1d(block_size, "scan_alloc_scratch");
  ScanScratch scratch;
  const std::size_t n_bytes = static_cast<std::size_t>(block_size) * sizeof(float);
  KF_CUDA_CHECK(cudaMalloc(&scratch.d_block_sums, n_bytes));
  KF_CUDA_CHECK(cudaMalloc(&scratch.d_block_sums_scanned, n_bytes));
  KF_CUDA_CHECK(cudaMalloc(&scratch.d_block_offsets, n_bytes));
  KF_CUDA_CHECK(cudaMalloc(&scratch.d_level2_dummy_sums, sizeof(float)));
  return scratch;
}

void scan_free_scratch(ScanScratch& scratch) {
  if (scratch.d_block_sums) cudaFree(scratch.d_block_sums);
  if (scratch.d_block_sums_scanned) cudaFree(scratch.d_block_sums_scanned);
  if (scratch.d_block_offsets) cudaFree(scratch.d_block_offsets);
  if (scratch.d_level2_dummy_sums) cudaFree(scratch.d_level2_dummy_sums);
  scratch.d_block_sums = nullptr;
  scratch.d_block_sums_scanned = nullptr;
  scratch.d_block_offsets = nullptr;
  scratch.d_level2_dummy_sums = nullptr;
}

void scan_multilevel_launch(BlockScanPassFn block_scan_pass, const float* d_in, float* d_out,
                             const ScanScratch& scratch, std::size_t n, int block_size,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  validate_block_size_pow2_1d(block_size, "scan_multilevel_launch");
  out_block = dim3(static_cast<unsigned>(block_size));

  if (n == 0) {
    out_grid = dim3(0);
    return;
  }

  const std::size_t num_blocks =
      (n + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size);
  out_grid = dim3(static_cast<unsigned>(num_blocks));

  if (num_blocks > static_cast<std::size_t>(block_size)) {
    throw std::runtime_error(
        "kernelforge::scan_multilevel_launch: n=" + std::to_string(n) + " with block_size=" +
        std::to_string(block_size) + " requires " + std::to_string(num_blocks) +
        " blocks, exceeding this 2-level scan's supported ceiling of block_size^2 = " +
        std::to_string(static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size)) +
        " elements. A 3rd scan level would be required beyond that (see scan_common.cuh's "
        "\"Known, disclosed scope limit\" comment) -- not implemented.");
  }

  // --- Level 1: per-block inclusive scan of the full input. ---
  block_scan_pass(d_in, d_out, scratch.d_block_sums, n, block_size, stream);

  if (num_blocks == 1) return; // d_out already holds the final answer.

  // --- Level 2: scan the (<= block_size) array of block totals in ONE
  // block, so its own block dimension must be a power of two >= num_blocks. ---
  const int level2_block_size = next_pow2_at_least(num_blocks, kMinPow2BlockSize);
  block_scan_pass(scratch.d_block_sums, scratch.d_block_sums_scanned,
                   scratch.d_level2_dummy_sums, num_blocks, level2_block_size, stream);

  // --- Convert level 2's inclusive result into each block's exclusive carry-in. ---
  {
    const dim3 glue_block(static_cast<unsigned>(level2_block_size));
    const dim3 glue_grid(1);
    scan_compute_block_offsets_kernel<<<glue_grid, glue_block, 0, stream>>>(
        scratch.d_block_sums_scanned, scratch.d_block_sums, scratch.d_block_offsets, num_blocks);
    KF_CUDA_CHECK_LAST_ERROR();
  }

  // --- Level 3: broadcast each block's carry-in across its chunk. ---
  {
    const dim3 add_block(static_cast<unsigned>(block_size));
    const dim3 add_grid(static_cast<unsigned>(num_blocks));
    scan_add_block_offsets_kernel<<<add_grid, add_block, 0, stream>>>(d_out,
                                                                       scratch.d_block_offsets, n);
    KF_CUDA_CHECK_LAST_ERROR();
  }
}

} // namespace kernelforge::kernels
