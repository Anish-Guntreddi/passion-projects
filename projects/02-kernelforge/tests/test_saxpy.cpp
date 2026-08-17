// Correctness tests for SAXPY (FR6: randomized vs reference, edge sizes,
// deterministic seeds).
#include <stdexcept>
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/vector/saxpy.cuh"

namespace {

void check_size(float a, std::size_t n, std::uint64_t seed) {
  const std::vector<float> x = kernelforge::make_random_vector(n, seed);
  const std::vector<float> y = kernelforge::make_random_vector(n, seed + 1);
  std::vector<float> expected(n);
  kernelforge::reference::saxpy(a, x.data(), y.data(), expected.data(), n);

  std::vector<float> actual(n);
  kernelforge::kernels::saxpy_run_host(a, x.data(), y.data(), actual.data(), n);

  const auto cmp = kernelforge::allclose(actual.data(), expected.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}

} // namespace

KF_TEST(Saxpy, EmptyInputIsLegalAndTrivial) { check_size(2.0f, 0, kernelforge::kDefaultSeed); }

KF_TEST(Saxpy, SingleElement) { check_size(2.0f, 1, kernelforge::kDefaultSeed); }

KF_TEST(Saxpy, SmallNonPowerOfTwo) { check_size(-1.5f, 41, kernelforge::kDefaultSeed); }

KF_TEST(Saxpy, NonBlockMultiple) { check_size(3.25f, 257, kernelforge::kDefaultSeed); }

KF_TEST(Saxpy, LargeRandomized) {
  check_size(0.5f, 8 * 1024 * 1024 + 3, kernelforge::kDefaultSeed);
}

KF_TEST(Saxpy, ZeroScalarReducesToIdentityOnY) {
  const std::size_t n = 10'000;
  const std::vector<float> x = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);
  const std::vector<float> y =
      kernelforge::make_random_vector(n, kernelforge::kDefaultSeed + 1);
  std::vector<float> actual(n);
  kernelforge::kernels::saxpy_run_host(0.0f, x.data(), y.data(), actual.data(), n);
  const auto cmp = kernelforge::allclose(actual.data(), y.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}

KF_TEST(Saxpy, NegativeScalar) { check_size(-4.75f, 12'345, kernelforge::kDefaultSeed); }

KF_TEST(Saxpy, InvalidBlockSizeThrows) {
  const std::size_t n = 8;
  const std::vector<float> x = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);
  const std::vector<float> y =
      kernelforge::make_random_vector(n, kernelforge::kDefaultSeed + 1);
  std::vector<float> out(n);

  for (int bad_block_size : {0, -1, -256, 1025}) {
    bool threw = false;
    try {
      kernelforge::kernels::saxpy_run_host(2.0f, x.data(), y.data(), out.data(), n,
                                            bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}
