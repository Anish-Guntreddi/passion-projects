// Correctness tests for both transpose variants (V1 naive, V2 tiled).
// FR6: randomized vs reference, edge sizes (1x1, non-tile-multiple,
// rectangular, small/large), deterministic seeds. Both variants are
// checked against the SAME CPU reference on the SAME input, since the
// whole point of Phase 1 is that they must agree on correctness while
// differing (measurably, in benchmarks/) on performance.
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/transpose/transpose_naive.cuh"
#include "kernels/transpose/transpose_tiled.cuh"

namespace {

void check_size(int rows, int cols, std::uint64_t seed) {
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  const std::vector<float> in = kernelforge::make_random_vector(n, seed);
  std::vector<float> expected(n);
  kernelforge::reference::transpose(in.data(), expected.data(), rows, cols);

  std::vector<float> naive_out(n);
  kernelforge::kernels::transpose_naive_run_host(in.data(), naive_out.data(), rows, cols);
  const auto naive_cmp = kernelforge::allclose(naive_out.data(), expected.data(), n);
  KF_EXPECT_TRUE(naive_cmp.passed);

  std::vector<float> tiled_out(n);
  kernelforge::kernels::transpose_tiled_run_host(in.data(), tiled_out.data(), rows, cols);
  const auto tiled_cmp = kernelforge::allclose(tiled_out.data(), expected.data(), n);
  KF_EXPECT_TRUE(tiled_cmp.passed);
}

} // namespace

KF_TEST(Transpose, OneByOne) { check_size(1, 1, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, SingleRow) { check_size(1, 37, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, SingleColumn) { check_size(37, 1, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, SmallSquareNonTileMultiple) {
  check_size(17, 17, kernelforge::kDefaultSeed); // tile is 32x32; 17 < tile, forces a partial tile
}

KF_TEST(Transpose, SquareExactlyOneTile) { check_size(32, 32, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, SquareNonTileMultiple) {
  check_size(130, 130, kernelforge::kDefaultSeed); // 130 = 4*32 + 2: partial tile on both axes
}

KF_TEST(Transpose, RectangularWideShort) { check_size(64, 513, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, RectangularTallNarrow) { check_size(513, 64, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, LargeSquare) { check_size(2048, 2048, kernelforge::kDefaultSeed); }

KF_TEST(Transpose, NaiveAndTiledAgreeOnSameInput) {
  // Redundant with check_size's internal comparison, but stated explicitly
  // since this is the property Phase 1's headline claim depends on: the
  // two variants are numerically interchangeable, and only their measured
  // performance differs (see benchmarks/raw/transpose.jsonl).
  const int rows = 777, cols = 511;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const std::vector<float> in = kernelforge::make_random_vector(n, kernelforge::kDefaultSeed);

  std::vector<float> naive_out(n);
  std::vector<float> tiled_out(n);
  kernelforge::kernels::transpose_naive_run_host(in.data(), naive_out.data(), rows, cols);
  kernelforge::kernels::transpose_tiled_run_host(in.data(), tiled_out.data(), rows, cols);

  const auto cmp = kernelforge::allclose(naive_out.data(), tiled_out.data(), n);
  KF_EXPECT_TRUE(cmp.passed);
}
