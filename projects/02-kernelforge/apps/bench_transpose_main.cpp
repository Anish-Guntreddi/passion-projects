// Benchmark harness for the transpose kernel family (Phase 1). Selects
// between v1_naive and v2_tiled via --variant; both are exercised over the
// same sizes so their results are directly comparable (docs/optimization-
// method.md: "exactly one variable changes").
//
// Matrices are square (rows == cols == n) for this sweep so a single --n
// controls the whole problem size; the kernels themselves support
// rectangular matrices (see tests/test_transpose.cpp for non-square cases).
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
#include "kernels/transpose/transpose_naive.cuh"
#include "kernels/transpose/transpose_tiled.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;
} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_transpose: --n <square dim> is required\n");
    return 2;
  }
  // The transpose kernels index rows/cols as `int` (see
  // kernels/transpose/*.cu), but CliArgs::n is a `long long` (shared with
  // vector_add/saxpy, which use it as a std::size_t vector length and have
  // no such narrowing). A --n above INT_MAX would silently wrap/truncate
  // on the cast below, corrupting size math and launch geometry instead of
  // failing loudly (hard constraint 8).
  if (args.n > static_cast<long long>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr,
                 "bench_transpose: --n (%lld) exceeds INT_MAX; the transpose kernels index "
                 "rows/cols as `int` so dimensions must fit\n",
                 args.n);
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_transpose: %s\n", err->c_str());
    return 2;
  }
  const int rows = static_cast<int>(args.n);
  const int cols = static_cast<int>(args.n);
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

  const bool tiled = (args.variant == "v2_tiled");
  if (!tiled && args.variant != "v1_naive") {
    std::fprintf(stderr,
                 "bench_transpose: unknown --variant '%s' (expected v1_naive or v2_tiled)\n",
                 args.variant.c_str());
    return 2;
  }

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const std::vector<float> in = kernelforge::make_random_vector(n, args.seed);
    std::vector<float> ref(n);
    kernelforge::reference::transpose(in.data(), ref.data(), rows, cols);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(n);
    if (tiled) {
      kernelforge::kernels::transpose_tiled_run_host(in.data(), gpu_out.data(), rows, cols);
    } else {
      kernelforge::kernels::transpose_naive_run_host(in.data(), gpu_out.data(), rows, cols);
    }
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), n);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_transpose: CORRECTNESS FAILED (%s): %s\n",
                   args.variant.c_str(), cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
    KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      if (tiled) {
        kernelforge::kernels::transpose_tiled_launch(d_in, d_out, rows, cols, 0, grid, block);
      } else {
        kernelforge::kernels::transpose_naive_launch(d_in, d_out, rows, cols, 0, grid, block);
      }
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      if (tiled) {
        kernelforge::kernels::transpose_tiled_launch(d_in, d_out, rows, cols, 0, grid, block);
      } else {
        kernelforge::kernels::transpose_naive_launch(d_in, d_out, rows, cols, 0, grid, block);
      }
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "transpose";
    result.variant = args.variant;
    result.description =
        tiled ? "Shared-memory TILE x TILE staged transpose: coalesced global read AND "
                "write; unpadded shared tile (known bank-conflict, V4 padding not yet "
                "implemented)."
              : "Naive one-thread-per-element transpose: coalesced global read, "
                "pathological (stride=rows) global write.";
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = n;
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
    result.bytes_moved = static_cast<double>(n) * sizeof(float) * 2.0; // 1 read + 1 write per element
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 1 memory-access lab: naive vs shared-memory-tiled coalescing "
                   "comparison, square " +
                   std::to_string(rows) + "x" + std::to_string(cols) + " matrix.";

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "transpose %dx%d variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 rows, cols, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_transpose: FATAL: %s\n", e.what());
    return 1;
  }
}
