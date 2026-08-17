// Correctness tests for all three histogram/atomics variants (V1-V3), for
// FR6: edge sizes (0, 1, non-block-multiple, degenerate num_bins=1, large
// multi-block), deterministic seeds, and both contention profiles
// (kernelforge::ContentionProfile) -- see src/kernels/histogram/README.md.
// Histogram counts are integers: correctness requires an EXACT match
// against the reference, not a tolerance-based comparison.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/histogram/histogram_global_atomic.cuh"
#include "kernels/histogram/histogram_privatized.cuh"
#include "kernels/histogram/histogram_privatized_coarsened.cuh"

namespace {

void check_case(std::size_t n, int num_bins, kernelforge::ContentionProfile profile,
                 std::uint64_t seed, int block_size = 256, int coarsen = 8) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_histogram_input(n, seed, profile);
  std::vector<int> expected(static_cast<std::size_t>(num_bins));
  kernelforge::reference::histogram(in.data(), n, expected.data(), num_bins);

  std::vector<int> v1(static_cast<std::size_t>(num_bins));
  histogram_global_atomic_run_host(in.data(), n, v1.data(), num_bins, block_size);
  KF_EXPECT_TRUE(v1 == expected);

  std::vector<int> v2(static_cast<std::size_t>(num_bins));
  histogram_privatized_run_host(in.data(), n, v2.data(), num_bins, block_size);
  KF_EXPECT_TRUE(v2 == expected);

  std::vector<int> v3(static_cast<std::size_t>(num_bins));
  histogram_privatized_coarsened_run_host(in.data(), n, v3.data(), num_bins, block_size, coarsen);
  KF_EXPECT_TRUE(v3 == expected);
}

} // namespace

KF_TEST(Histogram, EmptyIsAllZero) {
  check_case(0, 256, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, SingleElement) {
  check_case(1, 256, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, DegenerateOneBin) {
  // Every element must land in bin 0 -- maximal contention by construction,
  // independent of the input distribution.
  check_case(10'007, 1, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, NumBinsNotPowerOfTwo) {
  check_case(50'021, 100, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, SmallNonBlockMultiple) {
  check_case(1'000, 64, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, ExactlyOneBlock) {
  check_case(256, 32, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed, 256);
}

KF_TEST(Histogram, LargeMultiBlockUniform) {
  check_case(2'000'003, 256, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, LargeMultiBlockSkewed) {
  check_case(2'000'003, 256, kernelforge::ContentionProfile::kSkewed, kernelforge::kDefaultSeed);
}

KF_TEST(Histogram, CoarsenFactorOneMatchesUncoarsened) {
  using namespace kernelforge::kernels;
  const std::size_t n = 300'007;
  const int num_bins = 128;
  const std::vector<float> in =
      kernelforge::make_histogram_input(n, kernelforge::kDefaultSeed, kernelforge::ContentionProfile::kUniform);
  std::vector<int> v2(static_cast<std::size_t>(num_bins));
  histogram_privatized_run_host(in.data(), n, v2.data(), num_bins, 256);
  std::vector<int> v3(static_cast<std::size_t>(num_bins));
  histogram_privatized_coarsened_run_host(in.data(), n, v3.data(), num_bins, 256, /*elems_per_thread=*/1);
  KF_EXPECT_TRUE(v2 == v3);
}

KF_TEST(Histogram, CoarsenFactorDoesNotEvenlyDivideChunk) {
  check_case(123'457, 96, kernelforge::ContentionProfile::kSkewed, kernelforge::kDefaultSeed, 128,
             /*coarsen=*/5);
}

KF_TEST(Histogram, BlockSizeSweep) {
  for (int block_size : {32, 64, 128, 512, 1024}) {
    check_case(50'011, 64, kernelforge::ContentionProfile::kUniform, kernelforge::kDefaultSeed,
               block_size);
  }
}

KF_TEST(Histogram, DifferentSeedsStillAgree) {
  check_case(20'003, 128, kernelforge::ContentionProfile::kSkewed, kernelforge::kDefaultSeed + 7);
  check_case(20'003, 128, kernelforge::ContentionProfile::kSkewed, kernelforge::kDefaultSeed + 99);
}

KF_TEST(Histogram, ReferenceNumBinsNotPositiveThrows) {
  // FR6/hard constraint 8: reference::histogram must fail loudly on
  // num_bins <= 0 rather than clamp to bin -1 and write out of bounds
  // (see src/common/reference_ops.hpp's doc comment on histogram()).
  const std::size_t n = 8;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  for (int bad_num_bins : {0, -1, -100}) {
    std::vector<int> out(1); // never actually written on the throwing path
    bool threw = false;
    try {
      kernelforge::reference::histogram(in.data(), n, out.data(), bad_num_bins);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(Histogram, NumBinsTooLargeForSharedMemThrows) {
  using namespace kernelforge::kernels;
  const std::size_t n = 16;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  const int huge_num_bins = 20'000; // 20000 * 4 bytes = 80000 bytes > 49152-byte shared mem limit
  std::vector<int> out(static_cast<std::size_t>(huge_num_bins));
  bool threw = false;
  try {
    histogram_privatized_run_host(in.data(), n, out.data(), huge_num_bins);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  KF_EXPECT_TRUE(threw);
}

KF_TEST(Histogram, SkewedInputConcentratesIntoFewBins) {
  // Sanity check on make_histogram_input itself: the skewed profile should
  // put a large majority of mass into a small minority of bins, unlike
  // uniform. This is what makes it a genuine high-contention input rather
  // than just "a different seed" -- see src/common/rng.hpp.
  const std::size_t n = 200'000;
  const int num_bins = 256;
  const std::vector<float> skewed_in =
      kernelforge::make_histogram_input(n, kernelforge::kDefaultSeed, kernelforge::ContentionProfile::kSkewed);
  std::vector<int> skewed_hist(static_cast<std::size_t>(num_bins));
  kernelforge::reference::histogram(skewed_in.data(), n, skewed_hist.data(), num_bins);

  const std::vector<float> uniform_in =
      kernelforge::make_histogram_input(n, kernelforge::kDefaultSeed, kernelforge::ContentionProfile::kUniform);
  std::vector<int> uniform_hist(static_cast<std::size_t>(num_bins));
  kernelforge::reference::histogram(uniform_in.data(), n, uniform_hist.data(), num_bins);

  const int skewed_max = *std::max_element(skewed_hist.begin(), skewed_hist.end());
  const int uniform_max = *std::max_element(uniform_hist.begin(), uniform_hist.end());
  KF_EXPECT_TRUE(skewed_max > uniform_max * 10); // skewed's hottest bin should dwarf uniform's
}

// make_histogram_input only ever generates values in [0, 1) (see
// src/common/rng.hpp), so none of the cases above exercise the clamp
// branches in histogram_bin_of (histogram_common.cuh:47) /
// kernelforge::reference::histogram for x < 0 or x >= 1, nor pin down
// behavior exactly AT the domain edges and at exact bin boundaries. Build
// an explicit input covering all of those instead of relying on random
// generation to hit them.
KF_TEST(Histogram, BoundaryAndOutOfRangeValuesClampCorrectly) {
  using namespace kernelforge::kernels;
  const int num_bins = 4; // bin_of(x) = clamp(trunc(x * 4), 0, 3)
  const std::vector<float> in = {
      // Below the [0, 1) domain -- must clamp to bin 0.
      -1.0f, -0.1f, -1e-6f,
      // The domain's inclusive lower edge.
      0.0f,
      // Exact bin boundaries (k/4 for k=1,2,3): land in bin k, not k-1.
      0.25f, 0.5f, 0.75f,
      // Just under the domain's upper edge -> still the last bin (3).
      0.999999f,
      // At and beyond the [0, 1) domain's exclusive upper edge -- must
      // clamp to bin num_bins - 1 (3), not overflow past it.
      1.0f, 1.1f, 2.0f, 1e6f,
  };
  std::vector<int> expected(static_cast<std::size_t>(num_bins));
  kernelforge::reference::histogram(in.data(), in.size(), expected.data(), num_bins);
  // Pin the reference's own clamp behavior down explicitly, so this test
  // fails loudly (not just "GPU disagrees with an unverified reference")
  // if reference_ops.cpp's independently-maintained copy of the clamp
  // formula (see histogram_common.cuh's doc comment on why there are two)
  // ever drifts.
  KF_EXPECT_EQ(expected[0], 4); // -1.0f, -0.1f, -1e-6f, 0.0f
  KF_EXPECT_EQ(expected[1], 1); // 0.25f
  KF_EXPECT_EQ(expected[2], 1); // 0.5f
  KF_EXPECT_EQ(expected[3], 6); // 0.75f, 0.999999f, 1.0f, 1.1f, 2.0f, 1e6f

  std::vector<int> v1(static_cast<std::size_t>(num_bins));
  histogram_global_atomic_run_host(in.data(), in.size(), v1.data(), num_bins);
  KF_EXPECT_TRUE(v1 == expected);

  std::vector<int> v2(static_cast<std::size_t>(num_bins));
  histogram_privatized_run_host(in.data(), in.size(), v2.data(), num_bins);
  KF_EXPECT_TRUE(v2 == expected);

  std::vector<int> v3(static_cast<std::size_t>(num_bins));
  histogram_privatized_coarsened_run_host(in.data(), in.size(), v3.data(), num_bins);
  KF_EXPECT_TRUE(v3 == expected);
}

// FR6/hard constraint 8: fail loudly rather than silently overflow. Every
// histogram kernel accumulates per-bin counts in a 32-bit `int`; a single
// bin could in the worst case receive up to n increments (e.g. num_bins ==
// 1). This must be rejected before any device work happens, so this test
// calls the device-pointer `_launch` entry points directly with a huge n
// and null/dummy pointers instead of `_run_host`, which would otherwise
// need to actually allocate an n > INT_MAX-element buffer (~8+ GiB) to
// reach the same check.
KF_TEST(Histogram, NExceedingIntMaxThrows) {
  using namespace kernelforge::kernels;
  const std::size_t huge_n = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  dim3 grid, block;

  bool threw_v1 = false;
  try {
    histogram_global_atomic_launch(nullptr, nullptr, huge_n, /*num_bins=*/4, /*block_size=*/256,
                                    0, grid, block);
  } catch (const std::invalid_argument&) {
    threw_v1 = true;
  }
  KF_EXPECT_TRUE(threw_v1);

  bool threw_v2 = false;
  try {
    histogram_privatized_launch(nullptr, nullptr, huge_n, /*num_bins=*/4, /*block_size=*/256, 0,
                                 grid, block);
  } catch (const std::invalid_argument&) {
    threw_v2 = true;
  }
  KF_EXPECT_TRUE(threw_v2);

  bool threw_v3 = false;
  try {
    histogram_privatized_coarsened_launch(nullptr, nullptr, huge_n, /*num_bins=*/4,
                                           /*block_size=*/256, /*elems_per_thread=*/8, 0, grid,
                                           block);
  } catch (const std::invalid_argument&) {
    threw_v3 = true;
  }
  KF_EXPECT_TRUE(threw_v3);
}
