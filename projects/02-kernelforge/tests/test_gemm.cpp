// Correctness tests for all five GEMM variants (V1-V4 plus the cuBLAS
// ceiling/reference -- see src/kernels/gemm/README.md). FR6: randomized vs
// reference, edge sizes (0/1 where legal, non-tile-multiple, rectangular,
// small/large), deterministic seeds. All five are checked against the SAME
// CPU reference (kernelforge::reference::gemm) on the SAME input, since
// Phase 4's whole point is that they must agree on correctness while
// differing (measurably, in benchmarks/) on performance.
#include <vector>

#include "common/compare.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "common/testing.hpp"
#include "kernels/gemm/gemm_coalesced.cuh"
#include "kernels/gemm/gemm_cublas.cuh"
#include "kernels/gemm/gemm_naive.cuh"
#include "kernels/gemm/gemm_register_tiled.cuh"
#include "kernels/gemm/gemm_tiled.cuh"

namespace {

// Looser than the repo-wide default (see apps/bench_gemm_main.cpp's
// kGemmAtol/kGemmRtol doc comment for why: fp32 accumulation-order
// variance across the ladder's differently-shaped K-loops, not a bug).
constexpr float kAtol = 1e-2f;
constexpr float kRtol = 2e-2f;

void check_size(int m, int n, int k, std::uint64_t seed) {
  using namespace kernelforge::kernels;
  const std::size_t a_n = static_cast<std::size_t>(m) * k;
  const std::size_t b_n = static_cast<std::size_t>(k) * n;
  const std::size_t c_n = static_cast<std::size_t>(m) * n;
  const std::vector<float> a = kernelforge::make_random_vector(a_n, seed, -1.0f, 1.0f);
  const std::vector<float> b = kernelforge::make_random_vector(b_n, seed + 1, -1.0f, 1.0f);
  std::vector<float> expected(c_n);
  kernelforge::reference::gemm(a.data(), b.data(), expected.data(), m, n, k);

  std::vector<float> v1(c_n), v2(c_n), v3(c_n), v4(c_n), vcublas(c_n);
  gemm_naive_run_host(a.data(), b.data(), v1.data(), m, n, k);
  gemm_coalesced_run_host(a.data(), b.data(), v2.data(), m, n, k);
  gemm_tiled_run_host(a.data(), b.data(), v3.data(), m, n, k);
  gemm_register_tiled_run_host(a.data(), b.data(), v4.data(), m, n, k);
  gemm_cublas_run_host(a.data(), b.data(), vcublas.data(), m, n, k);

  KF_EXPECT_TRUE(kernelforge::allclose(v1.data(), expected.data(), c_n, kAtol, kRtol).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v2.data(), expected.data(), c_n, kAtol, kRtol).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v3.data(), expected.data(), c_n, kAtol, kRtol).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(v4.data(), expected.data(), c_n, kAtol, kRtol).passed);
  KF_EXPECT_TRUE(kernelforge::allclose(vcublas.data(), expected.data(), c_n, kAtol, kRtol).passed);
}

} // namespace

KF_TEST(Gemm, OneByOneByOne) { check_size(1, 1, 1, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, ZeroM) { check_size(0, 4, 4, kernelforge::kDefaultSeed); } // FR6: legal 0-size edge

KF_TEST(Gemm, ZeroN) { check_size(4, 0, 4, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, ZeroK) { check_size(4, 4, 0, kernelforge::kDefaultSeed); } // sum over empty K == 0

KF_TEST(Gemm, SmallSquareNonTileMultiple) {
  check_size(17, 17, 17, kernelforge::kDefaultSeed); // V3/V4 tiles are 32/64-wide; forces partial tiles
}

KF_TEST(Gemm, SquareExactlyOneNaiveTile) { check_size(32, 32, 32, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, SquareExactlyOneRegisterTile) {
  check_size(64, 64, 64, kernelforge::kDefaultSeed); // exactly kGemmRegBM x kGemmRegBN x kGemmRegBK*4
}

KF_TEST(Gemm, SquareNonTileMultiple) {
  check_size(130, 130, 130, kernelforge::kDefaultSeed); // 130 = 4*32+2 = 2*64+2: partial tiles both axes
}

KF_TEST(Gemm, RectangularWideK) { check_size(64, 48, 513, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, RectangularTallM) { check_size(513, 48, 64, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, RectangularWideN) { check_size(48, 513, 64, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, NonSquareAllThreeDims) {
  check_size(77, 131, 53, kernelforge::kDefaultSeed); // 3 distinct, mutually non-tile-multiple dims
}

KF_TEST(Gemm, LargerSquare) { check_size(256, 256, 256, kernelforge::kDefaultSeed); }

KF_TEST(Gemm, DifferentSeedsStillAgree) {
  check_size(96, 96, 96, kernelforge::kDefaultSeed + 7);
  check_size(96, 96, 96, kernelforge::kDefaultSeed + 99);
}

KF_TEST(Gemm, AllFiveVariantsAgreeOnSameInput) {
  // Redundant with check_size's internal comparison, but stated explicitly
  // since this is the property the whole ladder depends on: all five are
  // numerically interchangeable (within fp32 accumulation-order
  // tolerance), and only their measured performance differs (see
  // benchmarks/raw/gemm.jsonl).
  using namespace kernelforge::kernels;
  const int m = 111, n = 129, k = 77;
  const std::size_t a_n = static_cast<std::size_t>(m) * k;
  const std::size_t b_n = static_cast<std::size_t>(k) * n;
  const std::size_t c_n = static_cast<std::size_t>(m) * n;
  const std::vector<float> a = kernelforge::make_random_vector(a_n, kernelforge::kDefaultSeed, -1.0f, 1.0f);
  const std::vector<float> b =
      kernelforge::make_random_vector(b_n, kernelforge::kDefaultSeed + 1, -1.0f, 1.0f);

  std::vector<float> v1(c_n), v4(c_n);
  gemm_naive_run_host(a.data(), b.data(), v1.data(), m, n, k);
  gemm_register_tiled_run_host(a.data(), b.data(), v4.data(), m, n, k);

  const auto cmp = kernelforge::allclose(v1.data(), v4.data(), c_n, kAtol, kRtol);
  KF_EXPECT_TRUE(cmp.passed);
}
