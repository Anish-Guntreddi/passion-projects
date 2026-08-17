// Unit tests for common/launch_validate.hpp's validate_block_size_1d, pure
// host-side logic with no GPU launch involved. Covers the fixer's fix for
// the reported issue: block_size=0 dividing by zero in host grid-size
// arithmetic, negative block_size sign-converting into a huge unsigned
// dim3::x, and oversized block_size only being caught late by the CUDA
// runtime -- all three must now be rejected here, before any of that
// arithmetic runs.
#include <stdexcept>

#include "common/launch_validate.hpp"
#include "common/testing.hpp"

namespace {

bool throws_invalid_argument(int block_size) {
  try {
    kernelforge::validate_block_size_1d(block_size, "test_launcher");
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

} // namespace

KF_TEST(LaunchValidate, ZeroBlockSizeThrows) {
  KF_EXPECT_TRUE(throws_invalid_argument(0));
}

KF_TEST(LaunchValidate, NegativeBlockSizeThrows) {
  KF_EXPECT_TRUE(throws_invalid_argument(-1));
  KF_EXPECT_TRUE(throws_invalid_argument(-256));
}

KF_TEST(LaunchValidate, OverMaxBlockSizeThrows) {
  KF_EXPECT_TRUE(throws_invalid_argument(kernelforge::kMaxThreadsPerBlock1D + 1));
  KF_EXPECT_TRUE(throws_invalid_argument(4096));
}

KF_TEST(LaunchValidate, BoundaryAndTypicalValuesDoNotThrow) {
  for (int block_size : {1, 32, 64, 128, 256, 512, kernelforge::kMaxThreadsPerBlock1D}) {
    KF_EXPECT_TRUE(!throws_invalid_argument(block_size));
  }
}
