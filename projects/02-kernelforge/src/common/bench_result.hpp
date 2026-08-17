// Machine-readable benchmark result schema (FR3, FR4; ADR 0010).
//
// This header is the single source of truth for what every committed
// benchmark result records. The companion JSON Schema document at
// benchmarks/schema/bench_result.schema.json describes the same shape for
// non-C++ readers/validators (scripts/validate_results.py checks the two
// stay in sync in practice, by validating every emitted record against it).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/device_query.hpp"
#include "common/stats.hpp"

namespace kernelforge {

// Everything needed to know *what machine and toolchain* produced a result
// (FR4: "GPU model + compute capability; CUDA/toolkit/compiler versions;
// clock/power caveats").
struct EnvironmentInfo {
  std::string gpu_name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::string cuda_driver_version;   // e.g. "13.1"
  std::string cuda_runtime_version;  // e.g. "12.6"
  std::string host_compiler_id;      // e.g. "GNU"
  std::string host_compiler_version; // e.g. "13.3.0"
  std::string cuda_compiler_id;      // e.g. "NVIDIA"
  std::string cuda_compiler_version; // e.g. "12.6.85"
  std::string cuda_architectures;    // e.g. "89"
  std::string build_type;            // e.g. "RelWithDebInfo"
  std::string compile_flags_note;

  // ADR 0005: locked clocks are not available on this WSL2 setup. This is
  // recorded per-result (not just in a doc) so no individual result can be
  // misread as having controlled clocks.
  bool locked_clocks = false;
  std::string clock_lock_note;

  // Observed (not locked) clocks at the time this result's process started,
  // via `nvidia-smi --query-gpu=clocks.sm,clocks.mem`. Empty if nvidia-smi
  // was unavailable.
  int observed_sm_clock_mhz = -1;
  int observed_mem_clock_mhz = -1;

  std::string os_note;
  std::string timestamp_utc;

  // Captures compile-time build_info + a runtime device query + an
  // nvidia-smi clock snapshot into one EnvironmentInfo.
  static EnvironmentInfo capture(const DeviceInfo& dev);
};

// One committed measurement of one (kernel_family, variant, size) triple.
struct BenchResult {
  std::string schema_version = "1.0";

  std::string kernel_family; // "vector_add" | "saxpy" | "transpose" | "stride_copy"
  std::string variant;       // "v1_naive" | "v2_tiled" | ...
  std::string description;

  EnvironmentInfo env;

  std::size_t n = 0; // primary element count (vector length; rows*cols for transpose)
  int rows = 0;       // transpose only, 0 otherwise
  int cols = 0;       // transpose only, 0 otherwise
  int stride = 0;      // stride_copy only, 0 otherwise
  std::string dtype = "fp32";

  int block_dim_x = 0, block_dim_y = 0, block_dim_z = 1;
  int grid_dim_x = 0, grid_dim_y = 0, grid_dim_z = 1;

  int warmup_iters = 0;
  int measured_reps = 0;
  std::uint64_t seed = 0;

  // Every individual repetition's kernel-only elapsed time (FR4: "median +
  // distribution, never a single timing"). finalize() fills `stats` from
  // this.
  std::vector<double> raw_timings_ms;
  TimingStats stats;

  // Useful bytes moved by the kernel (read + written), independent of
  // access pattern — this is what makes "effective bandwidth" meaningful
  // as a coalescing-quality signal (a pathological-stride kernel touches
  // the same number of useful bytes as a contiguous one, but takes far
  // longer, so its *effective* GB/s collapses).
  double bytes_moved = 0.0;
  double effective_bandwidth_gb_s = 0.0;

  // 0.0 for the purely memory-bound kernels in Phases 0-1 (FR4: "where
  // meaningful" — vector add/SAXPY/transpose/stride-copy do ~1 FLOP or 0
  // FLOPs per element moved, so FLOP/s is not an interesting number here;
  // GEMM/reduction in later phases will populate this).
  double gflops = 0.0;

  double tolerance_atol = 0.0;
  double tolerance_rtol = 0.0;
  bool correctness_passed = false;
  std::string correctness_note;

  std::string notes;
};

// Computes `result.stats` from `result.raw_timings_ms` and
// `result.effective_bandwidth_gb_s` from `result.bytes_moved` and the
// resulting median. Call after populating raw_timings_ms/bytes_moved and
// before to_json()/append_jsonl().
void finalize(BenchResult& result);

std::string to_json(const BenchResult& result);

// Appends one JSON line to `path` (creates the file/parent directory if
// needed). Never truncates — results accumulate across runs.
void append_jsonl(const std::string& path, const BenchResult& result);

} // namespace kernelforge
