// Dedicated coalescing/stride microbenchmark harness (Phase 1 exit
// criterion: "repo empirically demonstrates contiguous vs pathological
// access"). Sweeps --stride while holding --n (thread/element count)
// fixed; see kernels/microbench/stride_copy.cuh for the kernel itself.
//
// Correctness here is stronger than "touched elements match the
// reference": the output buffer is pre-filled with a sentinel value before
// the copy, so the check also proves the kernel does not write outside the
// `num_threads` indices it owns (out[i*stride] for i in [0, num_threads)).
#include <cstdio>
#include <exception>
#include <vector>

#include "bench_cli.hpp"
#include "common/bench_result.hpp"
#include "common/compare.hpp"
#include "common/device_query.hpp"
#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/rng.hpp"
#include "kernels/microbench/stride_copy.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;
constexpr std::size_t kDefaultNumThreads = 1ull << 20; // ~1M threads
constexpr float kSentinel = -999.0f;
} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.stride <= 0) {
    std::fprintf(stderr, "bench_stride: --stride <positive int> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_stride: %s\n", err->c_str());
    return 2;
  }
  const std::size_t num_threads =
      args.n > 0 ? static_cast<std::size_t>(args.n) : kDefaultNumThreads;
  const int stride = args.stride;

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const std::size_t buf_elems =
        kernelforge::kernels::stride_copy_buffer_elems(num_threads, stride);
    const std::vector<float> in = kernelforge::make_random_vector(buf_elems, args.seed);

    std::vector<float> expected(buf_elems, kSentinel);
    for (std::size_t t = 0; t < num_threads; ++t) {
      expected[t * static_cast<std::size_t>(stride)] = in[t * static_cast<std::size_t>(stride)];
    }

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(buf_elems, kSentinel);
    kernelforge::kernels::stride_copy_run_host(in.data(), gpu_out.data(), num_threads, stride,
                                                args.block_size);
    const auto cmp = kernelforge::allclose(gpu_out.data(), expected.data(), buf_elems);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_stride: CORRECTNESS FAILED (stride=%d): %s\n", stride,
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = buf_elems * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes));
    KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      kernelforge::kernels::stride_copy_launch(d_in, d_out, num_threads, stride,
                                                args.block_size, 0, grid, block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      kernelforge::kernels::stride_copy_launch(d_in, d_out, num_threads, stride,
                                                args.block_size, 0, grid, block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "stride_copy";
    result.variant = "stride_copy";
    result.description = "out[i*stride] = in[i*stride] for i in [0, num_threads); isolates "
                         "inter-thread memory distance as the single variable.";
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = num_threads;
    result.stride = stride;
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
    // Useful bytes moved is independent of stride BY DESIGN: exactly one
    // float read + one float written per thread, regardless of how far
    // apart those addresses are. This is what makes effective_bandwidth_gb_s
    // a clean coalescing signal.
    result.bytes_moved = static_cast<double>(num_threads) * sizeof(float) * 2.0;
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 1 dedicated coalescing/stride microbenchmark. num_threads=" +
                   std::to_string(num_threads) + ", stride=" + std::to_string(stride) +
                   ", buffer_elems=" + std::to_string(buf_elems) + ".";

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "stride_copy num_threads=%zu stride=%d: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 num_threads, stride, result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_stride: FATAL: %s\n", e.what());
    return 1;
  }
}
