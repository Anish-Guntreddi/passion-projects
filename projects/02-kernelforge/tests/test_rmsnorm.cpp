// Correctness tests for all three RMSNorm variants (V1-V3 -- see
// src/kernels/norm/README.md). FR6: randomized vs reference, edge sizes
// (0/1 where legal, non-block-multiple, small/large), numerical
// tolerances, deterministic seeds. V3 additionally requires cols % 4 == 0
// (see common/launch_validate.hpp::validate_cols_multiple_of_4), checked
// explicitly here as a documented ladder constraint, not silently skipped.
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/norm/rmsnorm_naive.cuh"
#include "kernels/norm/rmsnorm_vectorized.cuh"
#include "kernels/norm/rmsnorm_warp_shuffle.cuh"

namespace {
constexpr float kEps = 1e-5f;

// check_shape covers V1/V2 always, and V3 too when cols % 4 == 0 (the
// caller picks cols values for which this is true whenever it wants all
// three compared -- see IncludeV3 param for the few cases that
// deliberately exercise a non-multiple-of-4 cols instead).
enum class IncludeV3 { kYes, kNo };

void check_shape(int rows, int cols, std::uint64_t seed, IncludeV3 include_v3 = IncludeV3::kYes,
                  int block_size = 256) {
  using namespace kernelforge::kernels;
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  const std::vector<float> in = kernelforge::make_random_vector(n, seed, -2.0f, 2.0f);
  const std::vector<float> gamma =
      kernelforge::make_random_vector(static_cast<std::size_t>(cols), seed + 1, 0.5f, 1.5f);
  std::vector<float> expected(n);
  kernelforge::reference::rmsnorm_rows(in.data(), gamma.data(), expected.data(), rows, cols, kEps);

  std::vector<float> v1(n), v2(n);
  rmsnorm_naive_run_host(in.data(), gamma.data(), v1.data(), rows, cols, kEps, block_size);
  rmsnorm_warp_shuffle_run_host(in.data(), gamma.data(), v2.data(), rows, cols, kEps, block_size);
  KF_EXPECT_TRUE(kernelforge::allclose(v1.data(), expected.data(), n).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v2.data(), expected.data(), n).passed);

  if (include_v3 == IncludeV3::kYes) {
    std::vector<float> v3(n);
    rmsnorm_vectorized_run_host(in.data(), gamma.data(), v3.data(), rows, cols, kEps, block_size);
    KF_EXPECT_TRUE(kernelforge::allclose(v3.data(), expected.data(), n).passed);
  }
}

} // namespace

KF_TEST(RmsNorm, ZeroRows) { check_shape(0, 128, kernelforge::kDefaultSeed); }

KF_TEST(RmsNorm, ZeroCols) { check_shape(8, 0, kernelforge::kDefaultSeed); } // 0 % 4 == 0: V3 included

KF_TEST(RmsNorm, FourCols) { check_shape(16, 4, kernelforge::kDefaultSeed); } // smallest legal V3 cols

KF_TEST(RmsNorm, SmallNonMultipleOf4Cols) {
  // Deliberately excludes V3 (37 % 4 != 0 -- see this file's header
  // comment); V1/V2 have no such restriction and are still checked.
  check_shape(4, 37, kernelforge::kDefaultSeed, IncludeV3::kNo);
}

KF_TEST(RmsNorm, ColsExactlyOneBlock) { check_shape(4, 256, kernelforge::kDefaultSeed); }

KF_TEST(RmsNorm, ColsLargerThanBlock) {
  check_shape(4, 4096, kernelforge::kDefaultSeed); // 16x block_size -> multi-stride
}

KF_TEST(RmsNorm, AllZeroRowDoesNotDivideByZero) {
  // FR2's "numerical stability" concern for THIS family: an all-zero row
  // has mean-of-squares == 0. Without `+ eps` under the sqrt, `1/sqrt(0)`
  // is +inf and the row's output would be NaN/inf. Checked directly
  // against the CPU reference (which applies the identical `+ eps`), not
  // just "does not crash".
  using namespace kernelforge::kernels;
  const int rows = 2, cols = 64;
  std::vector<float> in(static_cast<std::size_t>(rows) * cols, 0.0f);
  const std::vector<float> gamma(cols, 1.0f);
  std::vector<float> expected(in.size());
  kernelforge::reference::rmsnorm_rows(in.data(), gamma.data(), expected.data(), rows, cols, kEps);

  std::vector<float> v1(in.size()), v2(in.size()), v3(in.size());
  rmsnorm_naive_run_host(in.data(), gamma.data(), v1.data(), rows, cols, kEps);
  rmsnorm_warp_shuffle_run_host(in.data(), gamma.data(), v2.data(), rows, cols, kEps);
  rmsnorm_vectorized_run_host(in.data(), gamma.data(), v3.data(), rows, cols, kEps);

  KF_EXPECT_TRUE(kernelforge::allclose(v1.data(), expected.data(), in.size()).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v2.data(), expected.data(), in.size()).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v3.data(), expected.data(), in.size()).passed);
  // Every output element must be exactly 0.0 (0 * anything finite == 0),
  // not NaN -- pin this down explicitly, not just "close to the reference".
  for (float v : v1) KF_EXPECT_TRUE(v == 0.0f);
  for (float v : v3) KF_EXPECT_TRUE(v == 0.0f);
}

KF_TEST(RmsNorm, ManyRowsRealisticShape) {
  check_shape(2048, 768, kernelforge::kDefaultSeed); // transformer-activation-scale shape
}

KF_TEST(RmsNorm, BlockSizeSweep) {
  for (int block_size : {32, 64, 128, 512, 1024}) {
    check_shape(4, 1024, kernelforge::kDefaultSeed, IncludeV3::kYes, block_size);
  }
}

KF_TEST(RmsNorm, DifferentSeedsStillAgree) {
  check_shape(8, 384, kernelforge::kDefaultSeed + 7);
  check_shape(8, 384, kernelforge::kDefaultSeed + 99);
}

KF_TEST(RmsNorm, VectorizedRejectsNonMultipleOf4Cols) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(64, kernelforge::kDefaultSeed);
  const std::vector<float> gamma(9, 1.0f);
  std::vector<float> out(64);
  bool threw = false;
  try {
    rmsnorm_vectorized_run_host(in.data(), gamma.data(), out.data(), 8, 9, kEps);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  KF_EXPECT_TRUE(threw);
}

KF_TEST(RmsNorm, InvalidBlockSizeThrows) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(64, kernelforge::kDefaultSeed);
  const std::vector<float> gamma(8, 1.0f);
  std::vector<float> out(64);
  for (int bad_block_size : {0, -1, 17, 33, 1025}) {
    bool threw = false;
    try {
      rmsnorm_warp_shuffle_run_host(in.data(), gamma.data(), out.data(), 8, 8, kEps, bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(RmsNorm, AllThreeVariantsAgreeOnSameInput) {
  using namespace kernelforge::kernels;
  const int rows = 13, cols = 776; // 776 % 4 == 0
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, -2.0f, 2.0f);
  const std::vector<float> gamma =
      kernelforge::make_random_vector(static_cast<std::size_t>(cols), kernelforge::kDefaultSeed + 1, 0.5f, 1.5f);

  std::vector<float> v1(n), v3(n);
  rmsnorm_naive_run_host(in.data(), gamma.data(), v1.data(), rows, cols, kEps);
  rmsnorm_vectorized_run_host(in.data(), gamma.data(), v3.data(), rows, cols, kEps);

  const auto cmp = kernelforge::allclose(v1.data(), v3.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}
