// Correctness tests for Phase 6's occupancy-report infrastructure (ADR
// 0014, src/common/occupancy.cuh). These are sanity checks on the CUDA
// Runtime occupancy-API numbers themselves (bounds, internal consistency,
// and agreement with this device's known limits from docs/gpu-model.md /
// DeviceInfo), NOT a re-test of any kernel's numerical correctness (that
// is each family's own test_*.cpp). Every kernel family that wires a
// `*_query_occupancy()` function (transpose, reduction, gemm -- the 3
// Phase 6 case-study families) is exercised here.
#include "common/device_query.hpp"
#include "common/occupancy.cuh"
#include "common/testing.hpp"
#include "kernels/gemm/gemm_register_tiled.cuh"
#include "kernels/gemm/gemm_tiled.cuh"
#include "kernels/reduction/reduce_sequential_addressing.cuh"
#include "kernels/reduction/reduce_vectorized_coarsened.cuh"
#include "kernels/reduction/reduce_warp_shuffle.cuh"
#include "kernels/transpose/transpose_naive.cuh"
#include "kernels/transpose/transpose_tiled.cuh"

namespace {

// Shared invariants every OccupancyReport must satisfy regardless of which
// kernel/launch config produced it -- these hold by construction from the
// CUDA occupancy API's own contract, so a violation means occupancy.cuh
// itself (not the kernel under test) has a bug.
void check_report_invariants(const kernelforge::OccupancyReport& r) {
  KF_EXPECT_TRUE(r.registers_per_thread > 0);
  KF_EXPECT_TRUE(r.max_threads_per_block_this_kernel > 0);
  KF_EXPECT_TRUE(r.max_threads_per_block_this_kernel <= 1024); // this GPU's hard limit
  KF_EXPECT_TRUE(r.device_sm_count > 0);
  KF_EXPECT_TRUE(r.device_max_threads_per_sm > 0);
  // The launch config queried is legal for this kernel (occupancy.cuh's
  // caller always passes a real, launchable block_size), so at least one
  // block must fit.
  KF_EXPECT_TRUE(r.max_active_blocks_per_sm >= 1);
  KF_EXPECT_TRUE(r.achieved_threads_per_sm == r.max_active_blocks_per_sm * r.block_size);
  KF_EXPECT_TRUE(r.achieved_threads_per_sm <= r.device_max_threads_per_sm);
  KF_EXPECT_TRUE(r.occupancy_fraction >= 0.0 && r.occupancy_fraction <= 1.0);
  KF_EXPECT_TRUE(r.suggested_block_size > 0);
  KF_EXPECT_TRUE(r.suggested_block_size <= 1024);
  KF_EXPECT_TRUE(r.suggested_min_grid_size > 0);
}

} // namespace

KF_TEST(Occupancy, TransposeNaiveReportIsSane) {
  check_report_invariants(kernelforge::kernels::transpose_naive_query_occupancy());
}

KF_TEST(Occupancy, TransposeTiledReportIsSane) {
  const auto r = kernelforge::kernels::transpose_tiled_query_occupancy();
  check_report_invariants(r);
  // transpose_tiled.cuh: kTransposeTileDim=32 -> 32*32=1024 threads/block,
  // 32*32*4=4096 static shared bytes (see transpose_tiled.cu).
  KF_EXPECT_EQ(r.block_size, 1024);
  KF_EXPECT_EQ(static_cast<int>(r.static_shared_bytes_per_block), 4096);
}

KF_TEST(Occupancy, ReductionVariantsAgreeOnBlockSizeAndSharedFormula) {
  // V2/V3/V4 all size their dynamic shared-memory allocation as
  // block_size * sizeof(float) (see each *.cu's *_query_occupancy()) --
  // this is the ONE variable each ladder rung holds constant while
  // changing the finalization strategy, so their OccupancyReports should
  // reflect the SAME dynamic_shared_bytes_per_block at the same block_size,
  // even if registers_per_thread differs between them.
  constexpr int block_size = 256;
  const auto v2 = kernelforge::kernels::reduce_sequential_addressing_query_occupancy(block_size);
  const auto v3 = kernelforge::kernels::reduce_warp_shuffle_query_occupancy(block_size);
  const auto v4 = kernelforge::kernels::reduce_vectorized_coarsened_query_occupancy(block_size);
  check_report_invariants(v2);
  check_report_invariants(v3);
  check_report_invariants(v4);

  KF_EXPECT_EQ(static_cast<int>(v2.dynamic_shared_bytes_per_block), block_size * 4);
  KF_EXPECT_EQ(static_cast<int>(v3.dynamic_shared_bytes_per_block), block_size * 4);
  KF_EXPECT_EQ(static_cast<int>(v4.dynamic_shared_bytes_per_block), block_size * 4);
  KF_EXPECT_TRUE(v2.block_size == block_size && v3.block_size == block_size &&
                 v4.block_size == block_size);
}

KF_TEST(Occupancy, ReductionQueryRespectsRequestedBlockSize) {
  // Occupancy must be queryable at any of this ladder's legal (power-of-2)
  // block sizes, not just the 256 default -- and block_size in the report
  // must echo exactly what was requested.
  for (int bs : {32, 64, 128, 256, 512, 1024}) {
    const auto r = kernelforge::kernels::reduce_warp_shuffle_query_occupancy(bs);
    check_report_invariants(r);
    KF_EXPECT_EQ(r.block_size, bs);
  }
}

KF_TEST(Occupancy, GemmTiledReportIsSane) {
  const auto r = kernelforge::kernels::gemm_tiled_query_occupancy();
  check_report_invariants(r);
  // gemm_tiled.cuh: kGemmTileDim=32 -> 1024 threads/block; a_tile+b_tile =
  // 2*32*32*4 = 8192 static shared bytes (see gemm_tiled.cu).
  KF_EXPECT_EQ(r.block_size, 1024);
  KF_EXPECT_EQ(static_cast<int>(r.static_shared_bytes_per_block), 8192);
}

KF_TEST(Occupancy, GemmRegisterTiledReportIsSane) {
  const auto r = kernelforge::kernels::gemm_register_tiled_query_occupancy();
  check_report_invariants(r);
  // gemm_register_tiled.cuh: kGemmRegBlockThreads = (64/8)*64 = 512.
  KF_EXPECT_EQ(r.block_size, 512);
  // a_tile (64*16) + b_tile (16*64), both fp32 -> 2*64*16*4 = 8192 bytes.
  KF_EXPECT_EQ(static_cast<int>(r.static_shared_bytes_per_block), 8192);
}

KF_TEST(Occupancy, GemmRegisterTiledUsesFewerOrEqualActiveBlocksPerSmThanTiled) {
  // V4 (register tiling) holds kGemmRegTM=8 fp32 accumulators per thread
  // plus more live state than V3's single accumulator -- register
  // pressure should be higher per thread. This does not by itself force
  // fewer active blocks/SM (occupancy also depends on shared memory and
  // thread-count limits, which differ between the two block shapes too),
  // but registers_per_thread itself must reflect V4's larger footprint.
  // See profiling/case-studies/03-gemm-register-tiling-occupancy.md for
  // the full measured comparison and interpretation.
  const auto v3 = kernelforge::kernels::gemm_tiled_query_occupancy();
  const auto v4 = kernelforge::kernels::gemm_register_tiled_query_occupancy();
  check_report_invariants(v3);
  check_report_invariants(v4);
  KF_EXPECT_TRUE(v4.registers_per_thread > v3.registers_per_thread);
}

KF_TEST(Occupancy, DeviceFieldsMatchQueryDeviceInfo) {
  // OccupancyReport's device_sm_count / device_max_threads_per_sm are
  // read straight from cudaGetDeviceProperties (occupancy.cuh) -- they
  // must agree exactly with common/device_query.hpp's own query of the
  // same device (ADR 0001: this repo targets exactly one GPU).
  const kernelforge::DeviceInfo info = kernelforge::query_device_info(0);
  const auto r = kernelforge::kernels::transpose_tiled_query_occupancy();
  KF_EXPECT_EQ(r.device_sm_count, info.multiprocessor_count);
  KF_EXPECT_EQ(r.device_max_threads_per_sm, info.max_threads_per_multiprocessor);
}
