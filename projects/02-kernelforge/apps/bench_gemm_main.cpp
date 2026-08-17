// Benchmark harness for the GEMM kernel family (Phase 4). Selects among
// V1-V4 plus the cuBLAS ceiling/reference via --variant; all are exercised
// over the same (M, N, K) so results are directly comparable (docs/
// optimization-method.md: "exactly one variable changes" -- see
// src/kernels/gemm/README.md for which variable each rung changes).
//
// Matrices default to square (M == N == K == --n) when --m/--k are not
// given, so a single --n controls the whole problem size for the main
// ladder sweep (same convention bench_transpose_main.cpp uses); pass
// --m/--n/--k explicitly for a rectangular shape.
#include <cstdio>
#include <exception>
#include <limits>
#include <vector>

#include "bench_cli.hpp"
#include "common/bench_result.hpp"
#include "common/compare.hpp"
#include "common/device_query.hpp"
#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "kernels/gemm/gemm_coalesced.cuh"
#include "kernels/gemm/gemm_cublas.cuh"
#include "kernels/gemm/gemm_naive.cuh"
#include "kernels/gemm/gemm_register_tiled.cuh"
#include "kernels/gemm/gemm_tiled.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;

// FP32 accumulation over up to a few thousand K-length dot products, with
// each rung's K-loop summing in a DIFFERENT order (V1/V2: sequential;
// V3/V4: tile-chunked, plus V4's B-value register reuse; cuBLAS: an
// internal, unknown-to-us order/algorithm), legitimately produces
// larger absolute differences from the double-accumulated CPU reference
// than the repo-wide default tolerance (1e-5 atol / 1e-4 rtol) was sized
// for (that default was chosen for O(1)-length accumulations -- see
// common/compare.hpp). A looser, still-meaningful GEMM-specific tolerance
// avoids papering over a real bug while not flagging normal fp32
// summation-order variance as one; see src/kernels/gemm/README.md
// "Correctness" for the empirical check that motivated this value.
constexpr float kGemmAtol = 1e-2f;
constexpr float kGemmRtol = 2e-2f;

using LaunchFn = void (*)(const float*, const float*, float*, int, int, int, cudaStream_t, dim3&,
                           dim3&);
using RunHostFn = float (*)(const float*, const float*, float*, int, int, int);

struct VariantOps {
  LaunchFn launch;
  RunHostFn run_host;
  const char* description;
};

VariantOps ops_for(const std::string& variant) {
  using namespace kernelforge::kernels;
  if (variant == "v1_naive") {
    return {&gemm_naive_launch, &gemm_naive_run_host,
            "One thread per output element, naive global-memory reads, no shared memory. "
            "Thread-to-output mapping deliberately leaves A reads and C writes uncoalesced "
            "(FR2's literal 'V1 one thread/output' rung)."};
  }
  if (variant == "v2_coalesced") {
    return {&gemm_coalesced_launch, &gemm_coalesced_run_host,
            "Same as V1 except which axis maps to threadIdx.x, making A reads and C writes "
            "coalesced. Only variable changed vs V1: the row/col <-> threadIdx.x/y mapping."};
  }
  if (variant == "v3_tiled") {
    return {&gemm_tiled_launch, &gemm_tiled_run_host,
            "Shared-memory tiling: A/B tiles staged through shared memory and reused by "
            "every thread in the block. Only variable changed vs V2: global-memory reuse "
            "via shared memory (still one thread per output)."};
  }
  if (variant == "v4_register_tiled") {
    return {&gemm_register_tiled_launch, &gemm_register_tiled_run_host,
            "Register tiling/coarsening: each thread computes 8 output elements sharing one "
            "column, reusing each shared-memory B read across all 8. Only variable changed "
            "vs V3: per-thread output count / shared-memory read reuse."};
  }
  if (variant == "cublas_sgemm_ceiling") {
    return {&gemm_cublas_launch, &gemm_cublas_run_host,
            "cuBLAS SGEMM. NOT a ladder rung -- a ceiling/reference measured for context "
            "only (spec FR2: 'cuBLAS is a ceiling/reference, never a target to beat'). "
            "See docs/decisions/0013-cublas-ceiling-methodology.md."};
  }
  throw std::invalid_argument(
      "unknown --variant '" + variant +
      "' (expected v1_naive, v2_coalesced, v3_tiled, v4_register_tiled, or "
      "cublas_sgemm_ceiling)");
}

} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_gemm: --n <N, or square M=N=K when --m/--k unset> is required\n");
    return 2;
  }
  // --m/--k default to --n (square sweep) when unset (-1).
  const long long m_ll = (args.m >= 0) ? args.m : args.n;
  const long long k_ll = (args.k >= 0) ? args.k : args.n;
  const long long n_ll = args.n;
  for (long long v : {m_ll, n_ll, k_ll}) {
    if (v > static_cast<long long>(std::numeric_limits<int>::max())) {
      std::fprintf(stderr,
                   "bench_gemm: an (M,N,K) dimension (%lld) exceeds INT_MAX; the GEMM kernels "
                   "index dimensions as `int`\n",
                   v);
      return 2;
    }
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_gemm: %s\n", err->c_str());
    return 2;
  }
  const int m = static_cast<int>(m_ll);
  const int n = static_cast<int>(n_ll);
  const int k = static_cast<int>(k_ll);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const VariantOps ops = ops_for(args.variant);

    const std::size_t a_n = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const std::size_t b_n = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    const std::size_t c_n = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    // Small magnitude range keeps K-term dot products from growing large
    // enough to make fp32 accumulation-order differences (see kGemmAtol/
    // kGemmRtol above) dominate the comparison at this repo's larger K
    // values, while still exercising real, nonzero, sign-varying data.
    const std::vector<float> a = kernelforge::make_random_vector(a_n, args.seed, -1.0f, 1.0f);
    const std::vector<float> b = kernelforge::make_random_vector(b_n, args.seed + 1, -1.0f, 1.0f);
    std::vector<float> ref(c_n);
    kernelforge::reference::gemm(a.data(), b.data(), ref.data(), m, n, k);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(c_n);
    ops.run_host(a.data(), b.data(), gpu_out.data(), m, n, k);
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), c_n, kGemmAtol, kGemmRtol);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_gemm: CORRECTNESS FAILED (%s): %s\n", args.variant.c_str(),
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t a_bytes = a_n * sizeof(float);
    const std::size_t b_bytes = b_n * sizeof(float);
    const std::size_t c_bytes = c_n * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_a, a_bytes > 0 ? a_bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_b, b_bytes > 0 ? b_bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_c, c_bytes > 0 ? c_bytes : sizeof(float)));
    if (a_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    if (b_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      ops.launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      ops.launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_a));
    KF_CUDA_CHECK(cudaFree(d_b));
    KF_CUDA_CHECK(cudaFree(d_c));

    kernelforge::BenchResult result;
    result.kernel_family = "gemm";
    result.variant = args.variant;
    result.description = ops.description;
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = c_n;
    result.rows = m; // M (ADR 0012: gemm reuses rows/cols for M/N)
    result.cols = n; // N
    result.k = k;
    result.dtype = "fp32";
    result.block_dim_x = static_cast<int>(block.x);
    result.block_dim_y = static_cast<int>(block.y);
    result.block_dim_z = static_cast<int>(block.z);
    result.grid_dim_x = static_cast<int>(grid.x);
    result.grid_dim_y = static_cast<int>(grid.y);
    result.grid_dim_z = static_cast<int>(grid.z);
    result.warmup_iters = args.warmup_iters;
    result.measured_reps = args.measured_reps;
    result.seed = args.seed;
    result.raw_timings_ms = timings;
    // Minimum unique working-set bytes (A + B + C), NOT actual DRAM
    // traffic (which varies hugely by rung -- that variation is exactly
    // what gflops/median_ms below is meant to capture for a compute-bound
    // kernel family; see FR4: "effective bandwidth OR FLOP/s where
    // meaningful").
    result.bytes_moved = static_cast<double>(a_n + b_n + c_n) * sizeof(float);
    result.tolerance_atol = kGemmAtol;
    result.tolerance_rtol = kGemmRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 4 GEMM ladder; M=" + std::to_string(m) + " N=" + std::to_string(n) +
                   " K=" + std::to_string(k) + ".";

    kernelforge::finalize(result);
    // GFLOP/s: 2*M*N*K (one multiply + one add per inner-product term),
    // the standard GEMM FLOP-counting convention -- computed after
    // finalize() so it can reuse the median.
    if (result.stats.median_ms > 0.0) {
      const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                            static_cast<double>(k);
      result.gflops = flops / (result.stats.median_ms * 1.0e6);
    }

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "gemm M=%d N=%d K=%d variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "gflops=%.2f  correctness=%s\n",
                 m, n, k, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.gflops, cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_gemm: FATAL: %s\n", e.what());
    return 1;
  }
}
