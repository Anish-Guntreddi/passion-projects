// Benchmark harness for the reduction kernel family (Phase 2). Selects
// among the four ladder rungs via --variant; all are exercised over the
// same sizes so results are directly comparable (docs/optimization-
// method.md: "exactly one variable changes" between adjacent rungs -- see
// src/kernels/reduction/README.md for which variable each rung changes).
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
#include "kernels/reduction/reduce_naive_global.cuh"
#include "kernels/reduction/reduce_naive_interleaved.cuh"
#include "kernels/reduction/reduce_sequential_addressing.cuh"
#include "kernels/reduction/reduce_vectorized_coarsened.cuh"
#include "kernels/reduction/reduce_warp_shuffle.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;

using LaunchFn = void (*)(const float*, float*, std::size_t, int, cudaStream_t, dim3&, dim3&, bool);
using RunHostFn = float (*)(const float*, std::size_t, float*, int);

struct VariantOps {
  LaunchFn launch;
  RunHostFn run_host;
  const char* description;
};

VariantOps ops_for(const std::string& variant) {
  using namespace kernelforge::kernels;
  if (variant == "v0_naive_global_atomic") {
    return {&reduce_naive_global_launch, &reduce_naive_global_run_host,
            "Pure global-memory baseline: no shared memory, every thread's single element "
            "combined via one global atomicAdd (FR2's literal 'naive global-memory' rung)."};
  }
  if (variant == "v1_naive_interleaved") {
    return {&reduce_naive_interleaved_launch, &reduce_naive_interleaved_run_host,
            "Interleaved shared-memory tree addressing + global atomicAdd combine "
            "(deliberately naive baseline)."};
  }
  if (variant == "v2_sequential_addressing") {
    return {&reduce_sequential_addressing_launch, &reduce_sequential_addressing_run_host,
            "Sequential (tid < stride) addressing; removes warp divergence vs V1. "
            "Only variable changed vs V1: the addressing pattern."};
  }
  if (variant == "v3_warp_shuffle") {
    return {&reduce_warp_shuffle_launch, &reduce_warp_shuffle_run_host,
            "Warp-shuffle (__shfl_down_sync, full mask) finalizes the last warp instead of "
            "shared memory. Only variable changed vs V2: the last-warp combine."};
  }
  if (variant == "v4_vectorized_coarsened") {
    return {&reduce_vectorized_coarsened_launch, &reduce_vectorized_coarsened_run_host,
            "float4 vectorized loads + grid-stride thread coarsening (capped block count). "
            "Only variable changed vs V3: the load phase."};
  }
  throw std::invalid_argument(
      "unknown --variant '" + variant +
      "' (expected v0_naive_global_atomic, v1_naive_interleaved, v2_sequential_addressing, "
      "v3_warp_shuffle, or v4_vectorized_coarsened)");
}

} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_reduction: --n <count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_reduction: %s\n", err->c_str());
    return 2;
  }
  const std::size_t n = static_cast<std::size_t>(args.n);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const VariantOps ops = ops_for(args.variant);

    const std::vector<float> in = kernelforge::make_random_vector(n, args.seed, 0.0f, 1.0f);
    const float ref = kernelforge::reference::reduce_sum(in.data(), n);

    // --- Correctness (before any timing) ---
    float gpu_sum = 0.0f;
    ops.run_host(in.data(), n, &gpu_sum, args.block_size);
    const auto cmp = kernelforge::allclose(&gpu_sum, &ref, 1);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_reduction: CORRECTNESS FAILED (%s): %s\n",
                   args.variant.c_str(), cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes > 0 ? bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    if (n > 0) {
      KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    }

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      // zero_output=true: warmup is untimed, so let the launch self-zero
      // d_out as it would for a standalone call (no methodology impact).
      ops.launch(d_in, d_out, n, args.block_size, 0, grid, block, /*zero_output=*/true);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // The launch helpers zero d_out (cudaMemsetAsync) as part of their
    // single-call contract; if that memset happened inside the timed
    // interval below, raw_timings_ms would measure memset+kernel instead
    // of kernel-only, contradicting FR4 and this file's own doc comment.
    // Zero it explicitly BEFORE starting the timer each rep (on the same
    // stream, so ordering is still guaranteed with no extra host sync
    // needed) and pass zero_output=false so the launch doesn't redo it
    // inside the measured span.
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      KF_CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), 0));
      kernelforge::GpuTimer timer;
      timer.start();
      ops.launch(d_in, d_out, n, args.block_size, 0, grid, block, /*zero_output=*/false);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "reduction";
    result.variant = args.variant;
    result.description = ops.description;
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = n;
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
    result.bytes_moved = static_cast<double>(n) * sizeof(float); // read-only reduction
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 2 reduction ladder; n=" + std::to_string(n) + ".";

    kernelforge::finalize(result);
    // GFLOP/s: reduction does ~n-1 adds total, regardless of ladder rung
    // (tree structure conserves total add count); computed after
    // finalize() so it can reuse the median.
    if (result.stats.median_ms > 0.0 && n > 1) {
      const double flops = static_cast<double>(n - 1);
      result.gflops = flops / (result.stats.median_ms * 1.0e6);
    }

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "reduction n=%zu variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  gflops=%.3f  correctness=%s\n",
                 n, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s, result.gflops,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_reduction: FATAL: %s\n", e.what());
    return 1;
  }
}
