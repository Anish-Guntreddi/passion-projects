// Benchmark harness for the vector_add kernel family (Phase 0/1).
//
// Flow (FR6: "correctness check runs before any timing"):
//   1. Query device, enforce capability (ADR 0002).
//   2. Generate deterministic input (common/rng.hpp).
//   3. Run ONE correctness pass (host round-trip vs CPU reference). If it
//      fails, print why and exit non-zero WITHOUT collecting any timing.
//   4. Allocate device buffers once; run warmup_iters untimed launches to
//      let clocks/caches settle, then measured_reps timed launches (CUDA
//      events, one BenchResult raw sample each).
//   5. Print one BenchResult JSON line to stdout (consumed by
//      scripts/bench_driver.py) and a human summary to stderr.
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
#include "kernels/vector/vector_add.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;
} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_vector_add: --n <count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_vector_add: %s\n", err->c_str());
    return 2;
  }
  const std::size_t n = static_cast<std::size_t>(args.n);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const std::vector<float> x = kernelforge::make_random_vector(n, args.seed);
    const std::vector<float> y = kernelforge::make_random_vector(n, args.seed + 1);
    std::vector<float> ref(n);
    kernelforge::reference::vector_add(x.data(), y.data(), ref.data(), n);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(n);
    kernelforge::kernels::vector_add_run_host(x.data(), y.data(), gpu_out.data(), n,
                                               args.block_size);
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), n);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_vector_add: CORRECTNESS FAILED: %s\n",
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = n * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_x, bytes));
    KF_CUDA_CHECK(cudaMalloc(&d_y, bytes));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
    KF_CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    KF_CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      kernelforge::kernels::vector_add_launch(d_x, d_y, d_out, n, args.block_size, 0, grid,
                                               block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      kernelforge::kernels::vector_add_launch(d_x, d_y, d_out, n, args.block_size, 0, grid,
                                               block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_x));
    KF_CUDA_CHECK(cudaFree(d_y));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "vector_add";
    result.variant = args.variant;
    result.description = "out[i] = x[i] + y[i]; one thread per element, unit-stride "
                          "coalesced read/write on both operands.";
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
    result.bytes_moved = static_cast<double>(n) * sizeof(float) * 3.0; // read x, read y, write out
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 0/1 baseline; single naive variant, unit-stride throughout.";

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "vector_add n=%zu variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 n, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_vector_add: FATAL: %s\n", e.what());
    return 1;
  }
}
