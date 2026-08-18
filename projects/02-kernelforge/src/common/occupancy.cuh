// Occupancy introspection (Phase 6: "Profiling & low-level analysis").
//
// This is the evidence source Phase 6 leans on in place of Nsight Compute
// hardware-counter metrics, which are genuinely unavailable in this
// environment (ERR_NVGPUCTRPERM under WSL2 -- see
// docs/decisions/0014-phase6-profiling-evidence-strategy.md). Every number
// this header produces comes from the CUDA Runtime's own occupancy API
// (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
// `cudaOccupancyMaxPotentialBlockSize`) and `cudaFuncGetAttributes`, applied
// to the ACTUAL compiled kernel (its real register/shared-memory footprint,
// not a guess) -- this is the CUDA driver computing theoretical occupancy
// from the cubin's own embedded resource-usage info plus this device's
// limits, a purely static/compile-time calculation. It is NOT gated by the
// GPU-performance-counter permission that blocks `ncu` and nsys's GPU-side
// (CUPTI activity) kernel trace in this environment, and it is NOT a
// measurement of *achieved* runtime occupancy (which would need the
// blocked hardware counters) -- it reports the *theoretical ceiling* the
// launch configuration allows, same as what `ncu`'s "Launch Statistics" /
// "Occupancy" sections would show for their theoretical-occupancy fields.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "common/error_check.cuh"

namespace kernelforge {

struct OccupancyReport {
  // Compiled kernel's own footprint (cudaFuncGetAttributes -- a property of
  // the cubin, independent of any particular launch).
  int registers_per_thread = 0;
  std::size_t static_shared_bytes_per_block = 0; // __shared__ arrays fixed at compile time
  int max_threads_per_block_this_kernel = 0;     // cudaFuncAttributes::maxThreadsPerBlock

  // The launch configuration actually queried against.
  int block_size = 0;
  std::size_t dynamic_shared_bytes_per_block = 0; // 3rd launch<<<>>> argument

  // cudaOccupancyMaxActiveBlocksPerMultiprocessor: how many blocks of this
  // exact shape can be co-resident on one SM simultaneously, given this
  // device's register file, shared-memory, and thread-count limits.
  int max_active_blocks_per_sm = 0;
  int achieved_threads_per_sm = 0; // max_active_blocks_per_sm * block_size
  int device_max_threads_per_sm = 0;
  double occupancy_fraction = 0.0; // achieved_threads_per_sm / device_max_threads_per_sm

  // cudaOccupancyMaxPotentialBlockSize: what block size the CUDA runtime
  // itself would have picked to maximize per-SM occupancy for this kernel,
  // for comparison against the block size this repo actually uses.
  int suggested_block_size = 0;
  int suggested_min_grid_size = 0;

  int device_sm_count = 0;
};

// `kernel` must be an actual __global__ function (its address, e.g.
// `my_kernel`, not a function pointer variable of unrelated type -- the
// template parameter deduces the correct `cudaFuncGetAttributes`/occupancy
// overload from it). Must be called from a translation unit where `kernel`
// is visible (i.e. from within the same .cu file that defines it), which is
// why every kernel family exposes a thin `*_query_occupancy()` wrapper next
// to its `*_launch()` function rather than this template being called
// directly from apps/.
template <typename KernelFunc>
OccupancyReport compute_occupancy(KernelFunc kernel, int block_size,
                                   std::size_t dynamic_shared_bytes, int device = 0) {
  OccupancyReport r;
  r.block_size = block_size;
  r.dynamic_shared_bytes_per_block = dynamic_shared_bytes;

  cudaFuncAttributes attrs{};
  KF_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  r.registers_per_thread = attrs.numRegs;
  r.static_shared_bytes_per_block = attrs.sharedSizeBytes;
  r.max_threads_per_block_this_kernel = attrs.maxThreadsPerBlock;

  cudaDeviceProp prop{};
  KF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  r.device_max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  r.device_sm_count = prop.multiProcessorCount;

  KF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &r.max_active_blocks_per_sm, kernel, block_size, dynamic_shared_bytes));
  r.achieved_threads_per_sm = r.max_active_blocks_per_sm * block_size;
  r.occupancy_fraction = r.device_max_threads_per_sm > 0
                              ? static_cast<double>(r.achieved_threads_per_sm) /
                                    static_cast<double>(r.device_max_threads_per_sm)
                              : 0.0;

  // Block-size-independent suggestion (dynamic shared memory held fixed at
  // 0 here since it varies with the caller's chosen block size for several
  // of this repo's kernels -- the suggestion is about block SIZE, so a
  // fixed reference point is used; see occupancy_report_main.cpp for how
  // this is presented alongside the block-size-specific query above).
  KF_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&r.suggested_min_grid_size,
                                                    &r.suggested_block_size, kernel, 0, 0));

  return r;
}

} // namespace kernelforge
