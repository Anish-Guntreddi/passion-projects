// Correctness tests for the stride/coalescing microbenchmark kernel.
// Verifies both that touched indices are copied correctly AND that
// untouched indices are left alone (no stray writes) across a range of
// strides, including edge sizes.
#include <stdexcept>
#include <vector>

#include "common/compare.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/microbench/stride_copy.cuh"

namespace {
constexpr float kSentinel = -999.0f;

void check_stride(std::size_t num_threads, int stride, std::uint64_t seed) {
  const std::size_t buf_elems =
      kernelforge::kernels::stride_copy_buffer_elems(num_threads, stride);
  const std::vector<float> in = kernelforge::make_random_vector(buf_elems, seed);

  std::vector<float> expected(buf_elems, kSentinel);
  for (std::size_t t = 0; t < num_threads; ++t) {
    expected[t * static_cast<std::size_t>(stride)] = in[t * static_cast<std::size_t>(stride)];
  }

  std::vector<float> actual(buf_elems, kSentinel);
  kernelforge::kernels::stride_copy_run_host(in.data(), actual.data(), num_threads, stride);

  const auto cmp = kernelforge::allclose(actual.data(), expected.data(), buf_elems);
  KF_EXPECT_TRUE(cmp.passed);
}

} // namespace

KF_TEST(StrideCopy, ZeroThreadsIsLegalAndTrivial) {
  check_stride(0, 1, kernelforge::kDefaultSeed);
}

KF_TEST(StrideCopy, SingleThreadAnyStride) { check_stride(1, 64, kernelforge::kDefaultSeed); }

KF_TEST(StrideCopy, ContiguousStrideOne) { check_stride(10'000, 1, kernelforge::kDefaultSeed); }

KF_TEST(StrideCopy, SmallStride) { check_stride(5'000, 4, kernelforge::kDefaultSeed); }

KF_TEST(StrideCopy, LargeStride) { check_stride(2'000, 128, kernelforge::kDefaultSeed); }

KF_TEST(StrideCopy, NonBlockMultipleThreadCount) {
  check_stride(1'000'003, 8, kernelforge::kDefaultSeed); // prime thread count
}

KF_TEST(StrideCopy, StrideSweepAllPassCorrectness) {
  for (int stride : {1, 2, 4, 8, 16, 32, 64}) {
    check_stride(4'096, stride, kernelforge::kDefaultSeed);
  }
}

KF_TEST(StrideCopy, InvalidBlockSizeThrows) {
  const std::size_t num_threads = 8;
  const int stride = 4;
  const std::size_t buf_elems =
      kernelforge::kernels::stride_copy_buffer_elems(num_threads, stride);
  const std::vector<float> in = kernelforge::make_random_vector(buf_elems, kernelforge::kDefaultSeed);
  std::vector<float> out(buf_elems, kSentinel);

  for (int bad_block_size : {0, -1, -256, 1025}) {
    bool threw = false;
    try {
      kernelforge::kernels::stride_copy_run_host(in.data(), out.data(), num_threads, stride,
                                                  bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}
