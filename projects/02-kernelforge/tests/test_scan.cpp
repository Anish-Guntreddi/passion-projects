// Correctness tests for both scan strategies (Hillis-Steele, Blelloch).
// FR6: randomized vs reference, edge sizes (0, 1, within one block,
// exactly one block, spanning multiple blocks, non-block-multiple, the
// 2-level scope ceiling), deterministic seeds. Both strategies are
// checked against the SAME CPU reference
// (kernelforge::reference::inclusive_scan) on the SAME input -- see
// src/kernels/scan/README.md.
#include <cstdint>
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/scan/scan_blelloch.cuh"
#include "kernels/scan/scan_hillis_steele.cuh"

namespace {

void check_size(std::size_t n, std::uint64_t seed, int block_size = 256) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(n, seed, 0.0f, 1.0f);
  std::vector<float> expected(n);
  kernelforge::reference::inclusive_scan(in.data(), expected.data(), n);

  std::vector<float> hs_out(n);
  scan_hillis_steele_run_host(in.data(), hs_out.data(), n, block_size);
  const auto hs_cmp = kernelforge::allclose(hs_out.data(), expected.data(), n);
  KF_EXPECT_TRUE(hs_cmp.passed);

  std::vector<float> bl_out(n);
  scan_blelloch_run_host(in.data(), bl_out.data(), n, block_size);
  const auto bl_cmp = kernelforge::allclose(bl_out.data(), expected.data(), n);
  KF_EXPECT_TRUE(bl_cmp.passed);
}

} // namespace

KF_TEST(Scan, EmptyIsEmpty) { check_size(0, kernelforge::kDefaultSeed); }

KF_TEST(Scan, SingleElement) { check_size(1, kernelforge::kDefaultSeed); }

KF_TEST(Scan, SmallWithinOneBlockNonPowerOfTwo) { check_size(37, kernelforge::kDefaultSeed); }

KF_TEST(Scan, ExactlyOneBlock) { check_size(256, kernelforge::kDefaultSeed); }

KF_TEST(Scan, OneMoreThanOneBlockTwoLevel) { check_size(257, kernelforge::kDefaultSeed); }

KF_TEST(Scan, OneLessThanOneBlock) { check_size(255, kernelforge::kDefaultSeed); }

KF_TEST(Scan, SpansManyBlocksNonBlockMultiple) {
  check_size(60'037, kernelforge::kDefaultSeed); // prime-ish, spans ~235 blocks of 256
}

KF_TEST(Scan, ExactlyAtTwoLevelCeiling) {
  check_size(65'536, kernelforge::kDefaultSeed, 256); // block_size^2 == 65536
}

KF_TEST(Scan, ExceedingTwoLevelCeilingThrows) {
  using namespace kernelforge::kernels;
  const std::size_t n = 65'537; // one past block_size^2 for block_size=256
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  std::vector<float> out(n);
  bool threw = false;
  try {
    scan_hillis_steele_run_host(in.data(), out.data(), n, 256);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  KF_EXPECT_TRUE(threw);
}

KF_TEST(Scan, MonotonicNonDecreasingForNonNegativeInput) {
  // Independent sanity check beyond the reference comparison: a prefix
  // sum of non-negative values must never decrease.
  using namespace kernelforge::kernels;
  const std::size_t n = 10'007;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  std::vector<float> out(n);
  scan_blelloch_run_host(in.data(), out.data(), n, 256);
  bool monotonic = true;
  for (std::size_t i = 1; i < n; ++i) {
    if (out[i] + 1e-4f < out[i - 1]) { // small slack for FP32 rounding
      monotonic = false;
      break;
    }
  }
  KF_EXPECT_TRUE(monotonic);
}

KF_TEST(Scan, BlockSizeSweep) {
  // n must fit every swept block_size's block_size^2 ceiling; the smallest
  // block_size tested here (32) caps n at 1024, so n=1000 is used for all
  // of them (see scan_common.cuh's "Known, disclosed scope limit").
  for (int block_size : {32, 64, 128, 512, 1024}) {
    check_size(1'000, kernelforge::kDefaultSeed, block_size);
  }
}

KF_TEST(Scan, InvalidBlockSizeThrows) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(16, kernelforge::kDefaultSeed);
  std::vector<float> out(16);
  for (int bad_block_size : {0, -1, 17, 33, 1025}) {
    bool threw = false;
    try {
      scan_hillis_steele_run_host(in.data(), out.data(), in.size(), bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(Scan, BothStrategiesAgreeOnSameInput) {
  using namespace kernelforge::kernels;
  const std::size_t n = 12'345;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  std::vector<float> hs_out(n), bl_out(n);
  scan_hillis_steele_run_host(in.data(), hs_out.data(), n);
  scan_blelloch_run_host(in.data(), bl_out.data(), n);
  const auto cmp = kernelforge::allclose(hs_out.data(), bl_out.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}
