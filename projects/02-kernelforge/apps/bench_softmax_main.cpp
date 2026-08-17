// Benchmark harness for the softmax kernel family (Phase 5). Selects
// among V1-V3 via --variant; all are exercised over the same (rows, cols)
// so results are directly comparable -- see src/kernels/softmax/README.md
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
#include "kernels/softmax/softmax_fused_online.cuh"
#include "kernels/softmax/softmax_naive.cuh"
#include "kernels/softmax/softmax_warp_shuffle.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;

using LaunchFn = void (*)(const float*, float*, int, int, int, cudaStream_t, dim3&, dim3&);
using RunHostFn = float (*)(const float*, float*, int, int, int);

struct VariantOps {
  LaunchFn launch;
  RunHostFn run_host;
  const char* description;
};

VariantOps ops_for(const std::string& variant) {
  using namespace kernelforge::kernels;
  if (variant == "v1_naive") {
    return {&softmax_naive_launch, &softmax_naive_run_host,
            "One block per row, shared-memory tree reductions for max then exp-sum, three "
            "full passes over each row. 'Simple CUDA first' baseline."};
  }
  if (variant == "v2_warp_shuffle") {
    return {&softmax_warp_shuffle_launch, &softmax_warp_shuffle_run_host,
            "Same three-pass structure as V1; block-wide max/sum reductions use warp-shuffle "
            "instead of a shared-memory tree. Only variable changed vs V1: reduction mechanism."};
  }
  if (variant == "v3_fused_online") {
    return {&softmax_fused_online_launch, &softmax_fused_online_run_host,
            "Online-softmax fusion: max and exp-sum computed together in one pass (two full "
            "passes total instead of three). Only variable changed vs V2: pass count / memory "
            "traffic (reduction mechanism -- warp shuffle -- carries over unchanged)."};
  }
  throw std::invalid_argument("unknown --variant '" + variant +
                               "' (expected v1_naive, v2_warp_shuffle, or v3_fused_online)");
}

} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_softmax: --n <cols> is required\n");
    return 2;
  }
  if (args.rows < 0) {
    std::fprintf(stderr, "bench_softmax: --rows <row count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_softmax: %s\n", err->c_str());
    return 2;
  }
  const int rows = static_cast<int>(args.rows);
  const int cols = static_cast<int>(args.n);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const VariantOps ops = ops_for(args.variant);

    const std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    // A representative-magnitude range for activation-like input (not
    // adversarially large, since numerical stability is validated
    // separately -- see tests/test_softmax.cpp's LargeMagnitudeInput case).
    const std::vector<float> in = kernelforge::make_random_vector(elems, args.seed, -10.0f, 10.0f);
    std::vector<float> ref(elems);
    kernelforge::reference::softmax_rows(in.data(), ref.data(), rows, cols);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(elems);
    ops.run_host(in.data(), gpu_out.data(), rows, cols, args.block_size);
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), elems);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_softmax: CORRECTNESS FAILED (%s): %s\n", args.variant.c_str(),
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = elems * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes > 0 ? bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes > 0 ? bytes : sizeof(float)));
    if (bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      ops.launch(d_in, d_out, rows, cols, args.block_size, 0, grid, block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      ops.launch(d_in, d_out, rows, cols, args.block_size, 0, grid, block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "softmax";
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
    // Useful bytes moved: V1/V2 read the row 3x + write 1x; V3 reads 2x +
    // writes 1x -- this genuinely differs per rung (that IS the mechanism
    // V3 changes), so it is computed per-variant rather than as one
    // formula shared across all three.
    const double read_passes = (args.variant == "v3_fused_online") ? 2.0 : 3.0;
    result.bytes_moved = static_cast<double>(elems) * sizeof(float) * (read_passes + 1.0);
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 5 softmax ladder; rows=" + std::to_string(rows) +
                   " cols=" + std::to_string(cols) + ".";

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "softmax rows=%d cols=%d variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 rows, cols, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_softmax: FATAL: %s\n", e.what());
    return 1;
  }
}
