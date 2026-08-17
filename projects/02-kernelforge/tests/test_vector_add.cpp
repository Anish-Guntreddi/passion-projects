// Correctness tests for the vector_add kernel (Phase 0 exit criterion:
// "vector add validates"). FR6: randomized vs reference, edge sizes
// (0/1, non-power-of-two, non-block-multiple, small/large), deterministic
// seeds, tolerance-based comparison.
#include <stdexcept>
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/vector/vector_add.cuh"

namespace {

void check_size(std::size_t n, std::uint64_t seed) {
  const std::vector<float> x = kernelforge::make_random_vector(n, seed);
  const std::vector<float> y = kernelforge::make_random_vector(n, seed + 1);
  std::vector<float> expected(n);
  kernelforge::reference::vector_add(x.data(), y.data(), expected.data(), n);

  std::vector<float> actual(n);
  kernelforge::kernels::vector_add_run_host(x.data(), y.data(), actual.data(), n);

  const auto cmp = kernelforge::allclose(actual.data(), expected.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}

} // namespace

KF_TEST(VectorAdd, EmptyInputIsLegalAndTrivial) {
  check_size(0, kernelforge::kDefaultSeed);
}

KF_TEST(VectorAdd, SingleElement) {
  check_size(1, kernelforge::kDefaultSeed);
}

KF_TEST(VectorAdd, SmallNonPowerOfTwo) {
  check_size(37, kernelforge::kDefaultSeed);
}

KF_TEST(VectorAdd, ExactlyOneBlock256) {
  check_size(256, kernelforge::kDefaultSeed);
}

KF_TEST(VectorAdd, OneMoreThanOneBlock) {
  check_size(257, kernelforge::kDefaultSeed); // non-block-multiple: forces a partially-full block
}

KF_TEST(VectorAdd, MediumNonBlockMultiple) {
  check_size(1'000'003, kernelforge::kDefaultSeed); // prime, nowhere near a multiple of 256
}

KF_TEST(VectorAdd, LargeRandomized) {
  check_size(16 * 1024 * 1024 + 5, kernelforge::kDefaultSeed);
}

KF_TEST(VectorAdd, DeterministicAcrossRuns) {
  // Same seed -> same input -> same output, run twice, byte-for-byte.
  const std::size_t n = 10'000;
  const std::vector<float> x = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);
  const std::vector<float> y =
      kernelforge::make_random_vector(n, kernelforge::kDefaultSeed + 1);

  std::vector<float> out1(n);
  std::vector<float> out2(n);
  kernelforge::kernels::vector_add_run_host(x.data(), y.data(), out1.data(), n);
  kernelforge::kernels::vector_add_run_host(x.data(), y.data(), out2.data(), n);

  for (std::size_t i = 0; i < n; ++i) {
    KF_ASSERT_TRUE(out1[i] == out2[i]);
  }
}

KF_TEST(VectorAdd, InvalidBlockSizeThrows) {
  // Fixer regression test: block_size=0 used to divide by zero in host
  // grid-size arithmetic before any GPU code ran; negative block_size
  // sign-converted into a huge unsigned dim3::x. Both -- and an
  // oversized value -- must now throw std::invalid_argument instead.
  const std::size_t n = 8;
  const std::vector<float> x = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);
  const std::vector<float> y =
      kernelforge::make_random_vector(n, kernelforge::kDefaultSeed + 1);
  std::vector<float> out(n);

  for (int bad_block_size : {0, -1, -256, 1025}) {
    bool threw = false;
    try {
      kernelforge::kernels::vector_add_run_host(x.data(), y.data(), out.data(), n,
                                                 bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(VectorAdd, DifferentBlockSizesAgree) {
  const std::size_t n = 50'000;
  const std::vector<float> x = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);
  const std::vector<float> y =
      kernelforge::make_random_vector(n, kernelforge::kDefaultSeed + 1);
  std::vector<float> expected(n);
  kernelforge::reference::vector_add(x.data(), y.data(), expected.data(), n);

  for (int block_size : {32, 64, 128, 256, 512, 1024}) {
    std::vector<float> actual(n);
    kernelforge::kernels::vector_add_run_host(x.data(), y.data(), actual.data(), n, block_size);
    const auto cmp = kernelforge::allclose(actual.data(), expected.data(), n);
    KF_EXPECT_TRUE(cmp.passed);
  }
}
