// Benchmark harness for the RMSNorm kernel family (Phase 5). Selects
// among V1-V3 via --variant; all are exercised over the same (rows, cols)
// so results are directly comparable -- see src/kernels/norm/README.md
// for which variable each rung changes.
#include <cstdio>
#include <exception>
#include <vector>

#include "bench_cli.hpp"
#include "common/bench_result.hpp"
#include "common/compare.hpp"
#include "common/device_query.hpp"
#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "kernels/norm/rmsnorm_naive.cuh"
#include "kernels/norm/rmsnorm_vectorized.cuh"
#include "kernels/norm/rmsnorm_warp_shuffle.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;

using LaunchFn = void (*)(const float*, const float*, float*, int, int, float, int, cudaStream_t,
                           dim3&, dim3&);
using RunHostFn = float (*)(const float*, const float*, float*, int, int, float, int);

struct VariantOps {
  LaunchFn launch;
  RunHostFn run_host;
  const char* description;
};

VariantOps ops_for(const std::string& variant) {
  using namespace kernelforge::kernels;
  if (variant == "v1_naive") {
    return {&rmsnorm_naive_launch, &rmsnorm_naive_run_host,
            "One block per row, shared-memory tree reduction for sum-of-squares, two full "
            "passes over each row. 'Simple CUDA first' baseline."};
  }
  if (variant == "v2_warp_shuffle") {
    return {&rmsnorm_warp_shuffle_launch, &rmsnorm_warp_shuffle_run_host,
            "Same two-pass structure as V1; the sum-of-squares reduction uses warp-shuffle "
            "instead of a shared-memory tree. Only variable changed vs V1: reduction "
            "mechanism."};
  }
  if (variant == "v3_vectorized") {
    return {&rmsnorm_vectorized_launch, &rmsnorm_vectorized_run_host,
            "Same two-pass, warp-shuffle-reduced structure as V2; both passes load/store via "
            "float4 instead of one float at a time. Only variable changed vs V2: memory "
            "transaction width (requires cols % 4 == 0)."};
  }
  throw std::invalid_argument("unknown --variant '" + variant +
                               "' (expected v1_naive, v2_warp_shuffle, or v3_vectorized)");
}

} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_rmsnorm: --n <cols> is required\n");
    return 2;
  }
  if (args.rows < 0) {
    std::fprintf(stderr, "bench_rmsnorm: --rows <row count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_rmsnorm: %s\n", err->c_str());
    return 2;
  }
  const int rows = static_cast<int>(args.rows);
  const int cols = static_cast<int>(args.n);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const VariantOps ops = ops_for(args.variant);

    const std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::vector<float> in = kernelforge::make_random_vector(elems, args.seed, -1.0f, 1.0f);
    const std::vector<float> gamma =
        kernelforge::make_random_vector(static_cast<std::size_t>(cols), args.seed + 1, 0.5f, 1.5f);
    std::vector<float> ref(elems);
    kernelforge::reference::rmsnorm_rows(in.data(), gamma.data(), ref.data(), rows, cols, args.eps);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(elems);
    ops.run_host(in.data(), gamma.data(), gpu_out.data(), rows, cols, args.eps, args.block_size);
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), elems);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_rmsnorm: CORRECTNESS FAILED (%s): %s\n", args.variant.c_str(),
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = elems * sizeof(float);
    const std::size_t gamma_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    float* d_in = nullptr;
    float* d_gamma = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes > 0 ? bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_gamma, gamma_bytes > 0 ? gamma_bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes > 0 ? bytes : sizeof(float)));
    if (bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    if (gamma_bytes > 0) {
      KF_CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma_bytes, cudaMemcpyHostToDevice));
    }

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      ops.launch(d_in, d_gamma, d_out, rows, cols, args.eps, args.block_size, 0, grid, block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      ops.launch(d_in, d_gamma, d_out, rows, cols, args.eps, args.block_size, 0, grid, block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_gamma));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "rmsnorm";
    result.variant = args.variant;
    result.description = ops.description;
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = elems;
    result.rows = rows;
    result.cols = cols;
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
    // Useful bytes moved: every rung reads the row 2x + writes 1x (gamma
    // is tiny -- O(cols), not O(rows*cols) -- and omitted as negligible).
    result.bytes_moved = static_cast<double>(elems) * sizeof(float) * 3.0;
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 5 RMSNorm ladder; rows=" + std::to_string(rows) +
                   " cols=" + std::to_string(cols) + " eps=" + std::to_string(args.eps) + ".";

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "rmsnorm rows=%d cols=%d variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 rows, cols, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_rmsnorm: FATAL: %s\n", e.what());
    return 1;
  }
}
