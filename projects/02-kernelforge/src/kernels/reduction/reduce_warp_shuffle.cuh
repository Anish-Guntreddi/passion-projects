// Reduction V3 (warp-shuffle finalization). Exactly ONE variable changes
// versus reduce_sequential_addressing.cuh: how the LAST warp's worth of
// partial sums (32 remaining values) is combined. Everything else --
// block/grid size, the load phase, the down-to-32 sequential-addressing
// loop, the cross-block atomicAdd combine -- is identical to V2.
//
// HYPOTHESIS (written before this kernel existed):
// V2's tree reduction keeps using `__syncthreads()` + shared memory all
// the way down to a single surviving element, including the last 5 steps
// (stride 16, 8, 4, 2, 1) where all the remaining active threads already
// fit in one warp. On Volta and later (independent thread scheduling,
// sm_89 included -- ADR 0002), threads within a warp are NOT guaranteed
// to execute in lockstep, so those last shared-memory reads/writes are
// only safe here because V2 still pays for an explicit `__syncthreads()`
// at every step. ADR 0004 requires warp-level combines to instead use the
// `_sync` intrinsic family with an explicit, full participation mask.
// Replacing the last 5 shared-memory steps with `__shfl_down_sync` (full
// mask 0xffffffffu; this code path is only reached by tid < 32, which is
// warp-uniform, so full-mask participation is correct) trades 5
// `__syncthreads()` block-wide barriers + 5 rounds of shared-memory
// read/write for 5 register-to-register warp shuffles, which do not need
// a barrier and do not touch shared memory at all. Expect a modest but
// measurable speedup over V2, most visible at larger block sizes (more
// synchronization steps saved) and smaller problem sizes (relatively more
// of the kernel's total time is spent in this tail).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace kernelforge::kernels {

// `zero_output` (default true): see reduce_naive_interleaved.cuh -- pass
// false only when the caller already zeroed `*d_out` on the same stream.
void reduce_warp_shuffle_launch(const float* d_in, float* d_out, std::size_t n, int block_size,
                                 cudaStream_t stream, dim3& out_grid, dim3& out_block,
                                 bool zero_output = true);

float reduce_warp_shuffle_run_host(const float* h_in, std::size_t n, float* h_sum_out,
                                    int block_size = 256);

} // namespace kernelforge::kernels
