// Tiny CLI-argument parser shared by every bench_*_main.cpp. Supports both
// `--flag value` and `--flag=value` forms so it works equally well invoked
// by hand or from scripts/bench_driver.py.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

namespace kernelforge::apps {

// ADR 0005's benchmark-methodology floor: >= 20 measured repetitions
// (never a single timing, per FR4) and a non-negative warmup count. Also
// mirrored in benchmarks/schema/bench_result.schema.json's
// "measured_reps": {"minimum": 20}, which validate_results.py enforces on
// every *committed* result -- this check additionally fails fast at the
// CLI, before any GPU work runs, instead of only after a bad run has
// already been timed and printed.
constexpr int kMinMeasuredReps = 20;

struct CliArgs {
  std::string variant = "v1_naive";
  long long n = -1;      // vector_add / saxpy: vector length. transpose: square dim (rows=cols).
                          // reduction/scan: array length. histogram: input length.
                          // gemm: N (output cols); defaults to --m/--k when either is unset
                          // (square MxNxK sweep). softmax/rmsnorm: cols (reduction-axis length).
  long long m = -1;       // gemm only: M (output rows / A's row count). -1 => unset (see bench_gemm_main.cpp).
  long long k = -1;       // gemm only: K (A's col count / B's row count). -1 => unset.
  long long rows = -1;    // softmax/rmsnorm only: row (batch) count.
  int stride = -1;       // stride_copy only.
  int warmup_iters = 10; // ADR 0005 default.
  int measured_reps = 30; // ADR 0005 default (>= 20).
  std::uint64_t seed = 123456789ULL; // kernelforge::kDefaultSeed
  int block_size = 256;
  int num_bins = 256;             // histogram only.
  std::string contention = "uniform"; // histogram only: "uniform" | "skewed" (Phase 3, ADR 0011).
  int coarsen = 8;                // histogram v3_privatized_coarsened only: elements/thread.
  float eps = 1e-5f;              // rmsnorm only: numerical-stability epsilon added under the sqrt.
};

inline CliArgs parse_cli(int argc, char** argv) {
  CliArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next_value = [&](const char* flag) -> std::optional<std::string> {
      if (a == flag && i + 1 < argc) {
        return std::string(argv[i + 1]);
      }
      const std::string eq_prefix = std::string(flag) + "=";
      if (a.rfind(eq_prefix, 0) == 0) {
        return a.substr(eq_prefix.size());
      }
      return std::nullopt;
    };
    auto consume_if_separate = [&](const char* flag) {
      if (a == flag && i + 1 < argc) ++i;
    };

    if (auto v = next_value("--variant")) {
      args.variant = *v;
      consume_if_separate("--variant");
    } else if (auto v = next_value("--n")) {
      args.n = std::atoll(v->c_str());
      consume_if_separate("--n");
    } else if (auto v = next_value("--m")) {
      args.m = std::atoll(v->c_str());
      consume_if_separate("--m");
    } else if (auto v = next_value("--k")) {
      args.k = std::atoll(v->c_str());
      consume_if_separate("--k");
    } else if (auto v = next_value("--rows")) {
      args.rows = std::atoll(v->c_str());
      consume_if_separate("--rows");
    } else if (auto v = next_value("--eps")) {
      args.eps = std::strtof(v->c_str(), nullptr);
      consume_if_separate("--eps");
    } else if (auto v = next_value("--stride")) {
      args.stride = std::atoi(v->c_str());
      consume_if_separate("--stride");
    } else if (auto v = next_value("--warmup")) {
      args.warmup_iters = std::atoi(v->c_str());
      consume_if_separate("--warmup");
    } else if (auto v = next_value("--reps")) {
      args.measured_reps = std::atoi(v->c_str());
      consume_if_separate("--reps");
    } else if (auto v = next_value("--seed")) {
      args.seed = std::strtoull(v->c_str(), nullptr, 10);
      consume_if_separate("--seed");
    } else if (auto v = next_value("--block-size")) {
      args.block_size = std::atoi(v->c_str());
      consume_if_separate("--block-size");
    } else if (auto v = next_value("--num-bins")) {
      args.num_bins = std::atoi(v->c_str());
      consume_if_separate("--num-bins");
    } else if (auto v = next_value("--contention")) {
      args.contention = *v;
      consume_if_separate("--contention");
    } else if (auto v = next_value("--coarsen")) {
      args.coarsen = std::atoi(v->c_str());
      consume_if_separate("--coarsen");
    }
  }
  return args;
}

// Validates the ADR 0005 methodology floor (>= kMinMeasuredReps measured
// repetitions, warmup >= 0). Returns an error message if invalid,
// std::nullopt if OK. Every bench_*_main.cpp calls this immediately after
// parse_cli() and exits non-zero without touching the GPU on failure --
// this is a fail-fast CLI check, not a substitute for the committed-result
// schema check in scripts/validate_results.py (which still runs, and
// still owns rejecting anything that slips past this one).
inline std::optional<std::string> validate_methodology(const CliArgs& args) {
  if (args.measured_reps < kMinMeasuredReps) {
    return "--reps must be >= " + std::to_string(kMinMeasuredReps) +
           " (ADR 0005 / FR4: median + distribution, never a single timing); got " +
           std::to_string(args.measured_reps);
  }
  if (args.warmup_iters < 0) {
    return "--warmup must be >= 0; got " + std::to_string(args.warmup_iters);
  }
  return std::nullopt;
}

} // namespace kernelforge::apps
