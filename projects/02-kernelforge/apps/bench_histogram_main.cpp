// Benchmark harness for the histogram/atomics kernel family (Phase 3).
// Selects among the three contention-lab rungs via --variant and between
// the two input distributions via --contention -- see
// src/kernels/histogram/README.md for what each rung/distribution tests
// and benchmarks/methodology.md §10 for the as-run low- vs high-
// contention comparison this binary's sweeps produce.
//
// v3_privatized_coarsened takes an extra `elems_per_thread` (--coarsen)
// launch parameter the other two variants don't have, so this file
// dispatches with small local lambdas instead of a single uniform
// function-pointer type (the same intent as bench_transpose_main.cpp's
// boolean dispatch, generalized to three variants of differing arity).
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
#include "kernels/histogram/histogram_global_atomic.cuh"
#include "kernels/histogram/histogram_privatized.cuh"
#include "kernels/histogram/histogram_privatized_coarsened.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;
} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_histogram: --n <count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_histogram: %s\n", err->c_str());
    return 2;
  }
  const std::size_t n = static_cast<std::size_t>(args.n);
  const int num_bins = args.num_bins;

  const std::string& variant = args.variant;
  const bool is_v1 = (variant == "v1_global_atomic");
  const bool is_v2 = (variant == "v2_privatized");
  const bool is_v3 = (variant == "v3_privatized_coarsened");
  if (!is_v1 && !is_v2 && !is_v3) {
    std::fprintf(stderr,
                 "bench_histogram: unknown --variant '%s' (expected v1_global_atomic, "
                 "v2_privatized, or v3_privatized_coarsened)\n",
                 variant.c_str());
    return 2;
  }

  kernelforge::ContentionProfile profile;
  try {
    profile = kernelforge::contention_profile_from_string(args.contention);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_histogram: %s\n", e.what());
    return 2;
  }

  auto run_host = [&](const float* h_in, int* h_hist_out) -> float {
    using namespace kernelforge::kernels;
    if (is_v1) return histogram_global_atomic_run_host(h_in, n, h_hist_out, num_bins, args.block_size);
    if (is_v2) return histogram_privatized_run_host(h_in, n, h_hist_out, num_bins, args.block_size);
    return histogram_privatized_coarsened_run_host(h_in, n, h_hist_out, num_bins, args.block_size,
                                                    args.coarsen);
  };
  auto launch = [&](const float* d_in, int* d_hist, dim3& grid, dim3& block, bool zero_output) {
    using namespace kernelforge::kernels;
    if (is_v1) {
      histogram_global_atomic_launch(d_in, d_hist, n, num_bins, args.block_size, 0, grid, block,
                                      zero_output);
    } else if (is_v2) {
      histogram_privatized_launch(d_in, d_hist, n, num_bins, args.block_size, 0, grid, block,
                                   zero_output);
    } else {
      histogram_privatized_coarsened_launch(d_in, d_hist, n, num_bins, args.block_size,
                                             args.coarsen, 0, grid, block, zero_output);
    }
  };
  const char* description = is_v1 ? "Global-atomic baseline: every element's atomicAdd targets "
                                    "the global histogram directly."
                             : is_v2
                                 ? "Shared-memory privatization: block-private histogram, "
                                   "atomicAdd locally, merge once per block."
                                 : "Privatization + thread coarsening: each thread covers "
                                   "multiple elements before the block's single merge.";

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const std::vector<float> in = kernelforge::make_histogram_input(n, args.seed, profile);
    std::vector<int> ref(static_cast<std::size_t>(num_bins));
    kernelforge::reference::histogram(in.data(), n, ref.data(), num_bins);

    // --- Correctness (before any timing) ---
    std::vector<int> gpu_hist(static_cast<std::size_t>(num_bins));
    run_host(in.data(), gpu_hist.data());
    bool exact_match = (gpu_hist == ref);
    std::size_t first_mismatch = 0;
    if (!exact_match) {
      for (std::size_t b = 0; b < gpu_hist.size(); ++b) {
        if (gpu_hist[b] != ref[b]) {
          first_mismatch = b;
          break;
        }
      }
    }
    if (!exact_match) {
      std::fprintf(stderr,
                   "bench_histogram: CORRECTNESS FAILED (%s, contention=%s): bin %zu actual=%d "
                   "expected=%d (integer counts must match EXACTLY)\n",
                   variant.c_str(), args.contention.c_str(), first_mismatch,
                   gpu_hist[first_mismatch], ref[first_mismatch]);
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = n * sizeof(float);
    const std::size_t hist_bytes = static_cast<std::size_t>(num_bins) * sizeof(int);
    float* d_in = nullptr;
    int* d_hist = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes > 0 ? bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_hist, hist_bytes));
    if (n > 0) {
      KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    }

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      // zero_output=true: warmup is untimed, so let the launch self-zero
      // d_hist as it would for a standalone call (no methodology impact).
      launch(d_in, d_hist, grid, block, /*zero_output=*/true);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // The launch helpers zero d_hist (cudaMemsetAsync) as part of their
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
      KF_CUDA_CHECK(cudaMemsetAsync(d_hist, 0, hist_bytes, 0));
      kernelforge::GpuTimer timer;
      timer.start();
      launch(d_in, d_hist, grid, block, /*zero_output=*/false);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_hist));

    kernelforge::BenchResult result;
    result.kernel_family = "histogram";
    result.variant = variant;
    result.description = description;
    result.env = kernelforge::EnvironmentInfo::capture(device_info);
    result.n = n;
    result.num_bins = num_bins;
    result.contention_profile = kernelforge::to_string(profile);
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
    result.bytes_moved = static_cast<double>(n) * sizeof(float); // read-only input traffic
    result.tolerance_atol = 0.0; // integer counts: exact match required (see correctness check above)
    result.tolerance_rtol = 0.0;
    result.correctness_passed = exact_match;
    result.correctness_note = exact_match
                                   ? "exact integer bin-count match against reference::histogram"
                                   : "MISMATCH (should be unreachable: gated above)";
    result.notes = "Phase 3 atomics lab; n=" + std::to_string(n) +
                    ", num_bins=" + std::to_string(num_bins) +
                    ", contention=" + args.contention +
                    (is_v3 ? (", coarsen=" + std::to_string(args.coarsen)) : std::string());

    kernelforge::finalize(result);

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "histogram n=%zu num_bins=%d variant=%s contention=%s: median=%.4f ms  "
                 "IQR=[%.4f, %.4f] ms  effective_bw=%.2f GB/s  correctness=%s\n",
                 n, num_bins, variant.c_str(), args.contention.c_str(), result.stats.median_ms,
                 result.stats.p25_ms, result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 exact_match ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_histogram: FATAL: %s\n", e.what());
    return 1;
  }
}
