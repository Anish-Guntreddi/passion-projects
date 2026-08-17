// Correctness tests for all three softmax variants (V1-V3 -- see
// src/kernels/softmax/README.md). FR6: randomized vs reference, edge
// sizes (0/1 where legal, non-block-multiple, small/large), numerical
// tolerances, deterministic seeds, and -- specific to this family's FR2
// "numerical stability" ladder concern -- a large-magnitude input that
// would overflow float32 `exp()` without the max-subtraction every rung
// applies.
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/softmax/softmax_fused_online.cuh"
#include "kernels/softmax/softmax_naive.cuh"
#include "kernels/softmax/softmax_warp_shuffle.cuh"

namespace {

void check_shape(int rows, int cols, float lo, float hi, std::uint64_t seed, int block_size = 256) {
  using namespace kernelforge::kernels;
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  const std::vector<float> in = kernelforge::make_random_vector(n, seed, lo, hi);
  std::vector<float> expected(n);
  kernelforge::reference::softmax_rows(in.data(), expected.data(), rows, cols);

  std::vector<float> v1(n), v2(n), v3(n);
  softmax_naive_run_host(in.data(), v1.data(), rows, cols, block_size);
  softmax_warp_shuffle_run_host(in.data(), v2.data(), rows, cols, block_size);
  softmax_fused_online_run_host(in.data(), v3.data(), rows, cols, block_size);

  KF_EXPECT_TRUE(kernelforge::allclose(v1.data(), expected.data(), n).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v2.data(), expected.data(), n).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v3.data(), expected.data(), n).passed);
}

} // namespace

KF_TEST(Softmax, ZeroRows) { check_shape(0, 128, -1.0f, 1.0f, kernelforge::kDefaultSeed); }

KF_TEST(Softmax, ZeroCols) { check_shape(8, 0, -1.0f, 1.0f, kernelforge::kDefaultSeed); }

KF_TEST(Softmax, SingleElementRowsIsAlwaysOne) {
  // FR6 edge case: cols == 1 -> softmax is trivially 1.0 for every row
  // (exp(x-x)/exp(x-x) = 1), regardless of x's value.
  check_shape(16, 1, -5.0f, 5.0f, kernelforge::kDefaultSeed);
}

KF_TEST(Softmax, SmallNonPowerOfTwoCols) { check_shape(4, 37, -1.0f, 1.0f, kernelforge::kDefaultSeed); }

KF_TEST(Softmax, ColsExactlyOneBlock) { check_shape(4, 256, -1.0f, 1.0f, kernelforge::kDefaultSeed); }

KF_TEST(Softmax, ColsLargerThanBlock) {
  check_shape(4, 4096, -1.0f, 1.0f, kernelforge::kDefaultSeed); // 16x block_size -> multi-stride
}

KF_TEST(Softmax, ColsOneMoreThanBlockMultiple) {
  check_shape(3, 257, -1.0f, 1.0f, kernelforge::kDefaultSeed);
}

KF_TEST(Softmax, LargeMagnitudeInputStaysNumericallyStable) {
  // FR2's explicit "numerical stability" ladder concern, made concrete:
  // without max-subtraction, exp(1000) overflows float32 (max ~3.4e38,
  // exp(89) already exceeds it) before softmax is ever computed. Every
  // rung here subtracts the row max first, so this must still agree with
  // the (also max-subtracting) CPU reference to full default tolerance.
  check_shape(4, 512, -1000.0f, 1000.0f, kernelforge::kDefaultSeed);
}

KF_TEST(Softmax, ManyRowsRealisticShape) {
  check_shape(2048, 768, -3.0f, 3.0f, kernelforge::kDefaultSeed); // transformer-activation-scale shape
}

KF_TEST(Softmax, BlockSizeSweep) {
  for (int block_size : {32, 64, 128, 512, 1024}) {
    check_shape(4, 1000, -2.0f, 2.0f, kernelforge::kDefaultSeed, block_size);
  }
}

KF_TEST(Softmax, DifferentSeedsStillAgree) {
  check_shape(8, 384, -1.0f, 1.0f, kernelforge::kDefaultSeed + 7);
  check_shape(8, 384, -1.0f, 1.0f, kernelforge::kDefaultSeed + 99);
}

KF_TEST(Softmax, InvalidBlockSizeThrows) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(64, kernelforge::kDefaultSeed);
  std::vector<float> out(64);
  for (int bad_block_size : {0, -1, 17, 33, 1025}) {
    bool threw = false;
    try {
      softmax_warp_shuffle_run_host(in.data(), out.data(), 8, 8, bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(Softmax, AllThreeVariantsAgreeOnSameInput) {
  // Redundant with check_shape's internal comparison, but stated
  // explicitly since this is the property the whole ladder depends on.
  using namespace kernelforge::kernels;
  const int rows = 13, cols = 777;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, -4.0f, 4.0f);

  std::vector<float> v1(n), v3(n);
  softmax_naive_run_host(in.data(), v1.data(), rows, cols);
  softmax_fused_online_run_host(in.data(), v3.data(), rows, cols);

  const auto cmp = kernelforge::allclose(v1.data(), v3.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}
