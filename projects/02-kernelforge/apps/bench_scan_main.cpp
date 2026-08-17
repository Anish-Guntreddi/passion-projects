// Benchmark harness for the scan (inclusive prefix sum) kernel family
// (Phase 2). Selects between the two per-block strategies via --variant;
// both go through the identical multi-block orchestration
// (kernels/scan/scan_common.cuh) so only the per-block algorithm differs
// -- see src/kernels/scan/README.md.
//
// Known, disclosed scope limit (see kernels/scan/scan_common.cuh): n must
// be <= block_size^2. This binary reports that as a normal FATAL error
// (exit 1) rather than crashing, since scan_multilevel_launch throws
// std::runtime_error for an out-of-range n (hard constraint 8).
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#include "bench_cli.hpp"
#include "common/bench_result.hpp"
#include "common/compare.hpp"
#include "common/device_query.hpp"
#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"
#include "common/launch_validate.hpp"
#include "common/reference_ops.hpp"
#include "common/rng.hpp"
#include "kernels/scan/scan_blelloch.cuh"
#include "kernels/scan/scan_common.cuh"
#include "kernels/scan/scan_hillis_steele.cuh"

namespace {
constexpr int kRequiredMajor = 8;
constexpr int kRequiredMinor = 9;

// Per-block scan op count (additions only), by strategy -- the two
// strategies are NOT equivalent in op count (Hillis-Steele is the classic
// work-INEFFICIENT O(B log B) scan; Blelloch is work-EFFICIENT O(B)), so
// gflops must not reuse a single shared formula for both (that was this
// function's bug before it existed: a flat `flops = n` ignored this
// entirely). Derived directly from each kernel's loop structure:
//   - Hillis-Steele (scan_hillis_steele.cu): log2(B) passes, pass k has
//     (B - 2^k) active adds -> total = B*log2(B) - (B - 1).
//   - Blelloch (scan_blelloch.cu): B-1 up-sweep adds + B-1 down-sweep adds
//     (each a tree with B-1 total active nodes, the standard work-efficient
//     count) -> total = 2*(B - 1).
double per_block_scan_ops(const std::string& variant, int block_size) {
  const double b = static_cast<double>(block_size);
  if (variant == "hillis_steele") {
    return b * std::log2(b) - (b - 1.0);
  }
  return 2.0 * (b - 1.0); // blelloch
}

// Total additions across the whole (possibly 2-level) scan_multilevel_
// launch pipeline for this specific run, mirroring scan_common.cu's own
// control flow exactly (kMinPow2BlockSize/next_pow2_at_least's doubling-
// from-32 rule) so this count is the real op count for these grid/block
// dims, not an approximation:
//   - num_blocks == 1: just the one per-block scan pass (early return in
//     scan_multilevel_launch -- no level 2, no glue kernels).
//   - num_blocks > 1: level 1's `num_blocks` per-block scans, PLUS one
//     level-2 per-block scan (of the block-sums array, whose own block
//     size is the smallest power of two >= num_blocks starting from
//     kMinPow2BlockSize) to scan those block totals, PLUS `num_blocks`
//     subtracts (scan_compute_block_offsets_kernel) to turn that into
//     exclusive carries, PLUS `n` adds (scan_add_block_offsets_kernel) to
//     broadcast each block's carry across its chunk -- the last two glue
//     kernels are strategy-independent (kernels/scan/scan_common.cu).
double total_scan_ops(const std::string& variant, std::size_t n, int block_size,
                       std::size_t num_blocks) {
  if (n == 0) return 0.0;
  if (num_blocks <= 1) {
    return per_block_scan_ops(variant, block_size);
  }
  int level2_block_size = kernelforge::kMinPow2BlockSize;
  while (static_cast<std::size_t>(level2_block_size) < num_blocks) {
    level2_block_size <<= 1;
  }
  const double level1_ops = static_cast<double>(num_blocks) * per_block_scan_ops(variant, block_size);
  const double level2_ops = per_block_scan_ops(variant, level2_block_size);
  const double glue_ops = static_cast<double>(num_blocks) /* offset subtracts */ +
                           static_cast<double>(n) /* broadcast adds */;
  return level1_ops + level2_ops + glue_ops;
}

using LaunchFn = void (*)(const float*, float*, const kernelforge::kernels::ScanScratch&,
                           std::size_t, int, cudaStream_t, dim3&, dim3&);
using RunHostFn = float (*)(const float*, float*, std::size_t, int);

struct VariantOps {
  LaunchFn launch;
  RunHostFn run_host;
  const char* description;
};

VariantOps ops_for(const std::string& variant) {
  using namespace kernelforge::kernels;
  if (variant == "hillis_steele") {
    return {&scan_hillis_steele_launch, &scan_hillis_steele_run_host,
            "Hillis-Steele per-block scan: naive/step-efficient, every thread active every "
            "pass, log2(block_size) passes."};
  }
  if (variant == "blelloch") {
    return {&scan_blelloch_launch, &scan_blelloch_run_host,
            "Blelloch per-block scan: work-efficient up-sweep/down-sweep, "
            "2*log2(block_size) passes."};
  }
  throw std::invalid_argument("unknown --variant '" + variant +
                               "' (expected hillis_steele or blelloch)");
}

} // namespace

int main(int argc, char** argv) {
  using kernelforge::apps::parse_cli;
  const auto args = parse_cli(argc, argv);

  if (args.n < 0) {
    std::fprintf(stderr, "bench_scan: --n <count> is required\n");
    return 2;
  }
  if (auto err = kernelforge::apps::validate_methodology(args)) {
    std::fprintf(stderr, "bench_scan: %s\n", err->c_str());
    return 2;
  }
  const std::size_t n = static_cast<std::size_t>(args.n);

  try {
    const auto device_info = kernelforge::query_device_info(0);
    kernelforge::require_capability_or_throw(device_info, kRequiredMajor, kRequiredMinor);

    const VariantOps ops = ops_for(args.variant);

    const std::vector<float> in = kernelforge::make_random_vector(n, args.seed, 0.0f, 1.0f);
    std::vector<float> ref(n);
    kernelforge::reference::inclusive_scan(in.data(), ref.data(), n);

    // --- Correctness (before any timing) ---
    std::vector<float> gpu_out(n);
    ops.run_host(in.data(), gpu_out.data(), n, args.block_size);
    const auto cmp = kernelforge::allclose(gpu_out.data(), ref.data(), n);
    if (!cmp.passed) {
      std::fprintf(stderr, "bench_scan: CORRECTNESS FAILED (%s): %s\n", args.variant.c_str(),
                   cmp.summary().c_str());
      return 3;
    }

    // --- Timed sweep on persistent device buffers ---
    const std::size_t bytes = n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    KF_CUDA_CHECK(cudaMalloc(&d_in, bytes > 0 ? bytes : sizeof(float)));
    KF_CUDA_CHECK(cudaMalloc(&d_out, bytes > 0 ? bytes : sizeof(float)));
    if (n > 0) {
      KF_CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    }
    kernelforge::kernels::ScanScratch scratch =
        kernelforge::kernels::scan_alloc_scratch(args.block_size);

    dim3 grid, block;
    for (int i = 0; i < args.warmup_iters; ++i) {
      ops.launch(d_in, d_out, scratch, n, args.block_size, 0, grid, block);
      KF_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(args.measured_reps));
    for (int i = 0; i < args.measured_reps; ++i) {
      kernelforge::GpuTimer timer;
      timer.start();
      ops.launch(d_in, d_out, scratch, n, args.block_size, 0, grid, block);
      timings.push_back(static_cast<double>(timer.stop_ms()));
    }

    kernelforge::kernels::scan_free_scratch(scratch);
    KF_CUDA_CHECK(cudaFree(d_in));
    KF_CUDA_CHECK(cudaFree(d_out));

    kernelforge::BenchResult result;
    result.kernel_family = "scan";
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
    result.bytes_moved = static_cast<double>(n) * sizeof(float) * 2.0; // read in, write out
    result.tolerance_atol = kernelforge::kDefaultAtol;
    result.tolerance_rtol = kernelforge::kDefaultRtol;
    result.correctness_passed = cmp.passed;
    result.correctness_note = cmp.summary();
    result.notes = "Phase 2 scan strategy comparison; n=" + std::to_string(n) +
                    ", block_size=" + std::to_string(args.block_size) + ".";

    kernelforge::finalize(result);
    if (result.stats.median_ms > 0.0 && n > 0) {
      const double flops =
          total_scan_ops(args.variant, n, args.block_size, static_cast<std::size_t>(grid.x));
      result.gflops = flops / (result.stats.median_ms * 1.0e6);
    }

    std::printf("%s\n", kernelforge::to_json(result).c_str());
    std::fprintf(stderr,
                 "scan n=%zu variant=%s: median=%.4f ms  IQR=[%.4f, %.4f] ms  "
                 "effective_bw=%.2f GB/s  correctness=%s\n",
                 n, args.variant.c_str(), result.stats.median_ms, result.stats.p25_ms,
                 result.stats.p75_ms, result.effective_bandwidth_gb_s,
                 cmp.passed ? "PASS" : "FAIL");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_scan: FATAL: %s\n", e.what());
    return 1;
  }
}
