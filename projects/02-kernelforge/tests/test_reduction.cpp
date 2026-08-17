// Correctness tests for all five reduction ladder rungs (V0-V4, GPU side --
// V0 here is `reduce_naive_global`, the FR2 "naive global-memory" baseline;
// see its .cuh header for why it is a separate rung from V1's shared-memory
// tree). FR6: randomized vs reference, edge sizes (0, 1, non-power-of-two,
// non-block-multiple, small/large), deterministic seeds, a block-size
// sweep. All five variants are checked against the SAME CPU reference
// (kernelforge::reference::reduce_sum) on the SAME input, since Phase 2's
// whole point is that they must agree on correctness while differing
// (measurably, in benchmarks/) on performance -- see
// src/kernels/reduction/README.md.
#include <cstdint>
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/reduction/reduce_naive_global.cuh"
#include "kernels/reduction/reduce_naive_interleaved.cuh"
#include "kernels/reduction/reduce_sequential_addressing.cuh"
#include "kernels/reduction/reduce_vectorized_coarsened.cuh"
#include "kernels/reduction/reduce_warp_shuffle.cuh"

namespace {

// Non-negative inputs avoid catastrophic-cancellation edge cases (a sum
// near zero would make a relative tolerance meaningless) -- see
// src/common/reference_ops.hpp's reduce_sum doc comment.
void check_size(std::size_t n, std::uint64_t seed, int block_size = 256) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(n, seed, 0.0f, 1.0f);
  const float expected = kernelforge::reference::reduce_sum(in.data(), n);

  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;
  reduce_naive_global_run_host(in.data(), n, &v0, block_size);
  reduce_naive_interleaved_run_host(in.data(), n, &v1, block_size);
  reduce_sequential_addressing_run_host(in.data(), n, &v2, block_size);
  reduce_warp_shuffle_run_host(in.data(), n, &v3, block_size);
  reduce_vectorized_coarsened_run_host(in.data(), n, &v4, block_size);

  KF_EXPECT_TRUE(kernelforge::allclose(&v0, &expected, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v1, &expected, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v2, &expected, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v3, &expected, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v4, &expected, 1).passed);
}

} // namespace

KF_TEST(Reduction, EmptyIsZero) { check_size(0, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, SingleElement) { check_size(1, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, SmallNonPowerOfTwo) { check_size(37, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, ExactlyOneBlock) { check_size(256, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, OneMoreThanOneBlock) { check_size(257, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, OneLessThanOneBlock) { check_size(255, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, NonBlockMultiplePrime) {
  check_size(1'000'003, kernelforge::kDefaultSeed); // prime, not a multiple of any pow2 block
}

KF_TEST(Reduction, LargeMultiBlock) { check_size(4'194'304, kernelforge::kDefaultSeed); }

KF_TEST(Reduction, VeryLargeAwkwardSize) {
  check_size(16'777'213, kernelforge::kDefaultSeed); // 2^24 - 3
}

KF_TEST(Reduction, BlockSizeSweep) {
  for (int block_size : {32, 64, 128, 512, 1024}) {
    check_size(500'001, kernelforge::kDefaultSeed, block_size);
  }
}

KF_TEST(Reduction, DifferentSeedsStillAgree) {
  check_size(65'536, kernelforge::kDefaultSeed + 7);
  check_size(65'536, kernelforge::kDefaultSeed + 99);
}

KF_TEST(Reduction, InvalidBlockSizeThrows) {
  using namespace kernelforge::kernels;
  const std::vector<float> in = kernelforge::make_random_vector(16, kernelforge::kDefaultSeed);
  float out = 0.0f;
  for (int bad_block_size : {0, -1, 17, 33, 1025}) { // 17/33 are not powers of two
    bool threw = false;
    try {
      reduce_sequential_addressing_run_host(in.data(), in.size(), &out, bad_block_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    KF_EXPECT_TRUE(threw);
  }
}

KF_TEST(Reduction, NaiveGlobalAcceptsNonPow2BlockSize) {
  // V0 (reduce_naive_global) has no shared-memory tree step, so unlike
  // every other rung it must NOT require a power-of-two block_size -- only
  // reduce_naive_global.cuh's generic validate_block_size_1d bound applies.
  // This is the one deliberate difference in V0's launch contract vs the
  // rest of the ladder; pin it down explicitly.
  using namespace kernelforge::kernels;
  const std::size_t n = 10'007;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  const float expected = kernelforge::reference::reduce_sum(in.data(), n);
  for (int block_size : {17, 33, 100, 200}) { // deliberately non-power-of-two
    float out = 0.0f;
    reduce_naive_global_run_host(in.data(), n, &out, block_size);
    KF_EXPECT_TRUE(kernelforge::allclose(&out, &expected, 1).passed);
  }
}

KF_TEST(Reduction, AllFiveVariantsAgreeOnSameInput) {
  // Redundant with check_size's internal comparison, but stated explicitly
  // since this is the property the whole ladder depends on: all five
  // variants are numerically interchangeable, and only their measured
  // performance differs (see benchmarks/raw/reduction.jsonl).
  using namespace kernelforge::kernels;
  const std::size_t n = 777'001;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed, 0.0f, 1.0f);
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;
  reduce_naive_global_run_host(in.data(), n, &v0);
  reduce_naive_interleaved_run_host(in.data(), n, &v1);
  reduce_sequential_addressing_run_host(in.data(), n, &v2);
  reduce_warp_shuffle_run_host(in.data(), n, &v3);
  reduce_vectorized_coarsened_run_host(in.data(), n, &v4);

  KF_EXPECT_TRUE(kernelforge::allclose(&v0, &v1, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v1, &v2, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v2, &v3, 1).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(&v3, &v4, 1).passed);
}
