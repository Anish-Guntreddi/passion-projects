#include "common/bench_result.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "build_info.generated.hpp"

namespace kernelforge {

namespace {

std::string capture_timestamp_utc() {
  const std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
#if defined(_WIN32)
  gmtime_s(&tm_utc, &now);
#else
  gmtime_r(&now, &tm_utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return std::string(buf);
}

// Best-effort read of current (unlocked) SM/memory clocks via nvidia-smi.
// Returns {-1, -1} if nvidia-smi is unavailable or output is unparseable —
// this is a "nice to have" observability field, not load-bearing, so it
// fails soft rather than throwing.
std::pair<int, int> read_observed_clocks_mhz() {
  std::array<char, 256> buffer{};
  std::string output;
  // Explicit path avoids relying on PATH resolution differences between
  // interactive and non-interactive shells.
  FILE* pipe = popen(
      "/usr/lib/wsl/lib/nvidia-smi --query-gpu=clocks.sm,clocks.mem "
      "--format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (!pipe) {
    return {-1, -1};
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  pclose(pipe);

  int sm_clock = -1;
  int mem_clock = -1;
  const auto comma = output.find(',');
  if (comma != std::string::npos) {
    try {
      sm_clock = std::stoi(output.substr(0, comma));
      mem_clock = std::stoi(output.substr(comma + 1));
    } catch (...) {
      sm_clock = -1;
      mem_clock = -1;
    }
  }
  return {sm_clock, mem_clock};
}

void ensure_parent_dir(const std::string& path) {
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

std::string json_str(const std::string& s) { return "\"" + json_escape(s) + "\""; }

std::string json_bool(bool b) { return b ? "true" : "false"; }

std::string json_double_array(const std::vector<double>& v) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) oss << ",";
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}

} // namespace

EnvironmentInfo EnvironmentInfo::capture(const DeviceInfo& dev) {
  EnvironmentInfo env;
  env.gpu_name = dev.name;
  env.compute_capability_major = dev.compute_capability_major;
  env.compute_capability_minor = dev.compute_capability_minor;
  env.cuda_driver_version = format_cuda_version(dev.cuda_driver_version);
  env.cuda_runtime_version = format_cuda_version(dev.cuda_runtime_version);

  env.host_compiler_id = build_info::kCxxCompilerId;
  env.host_compiler_version = build_info::kCxxCompilerVersion;
  env.cuda_compiler_id = build_info::kCudaCompilerId;
  env.cuda_compiler_version = build_info::kCudaCompilerVersion;
  env.cuda_architectures = build_info::kCudaArchitectures;
  env.build_type = build_info::kBuildType;
  env.compile_flags_note =
      "nvcc: -lineinfo --expt-relaxed-constexpr, CMAKE_CUDA_ARCHITECTURES=" +
      env.cuda_architectures + ", CMAKE_BUILD_TYPE=" + env.build_type +
      " (RelWithDebInfo == -O2 -g for the host compiler); no --use_fast_math.";

  // ADR 0005: verified 2026-08-17 that this WSL2 guest cannot lock clocks.
  env.locked_clocks = false;
  env.clock_lock_note =
      "GPU clocks are NOT locked. `nvidia-smi -lgc ...` returns "
      "'The current user does not have permission to change clocks' under "
      "this WSL2 guest. See docs/decisions/0005-benchmark-noise-control.md. "
      "observed_sm_clock_mhz/observed_mem_clock_mhz below are read-only "
      "snapshots taken at result-capture time, not a controlled state.";

  const auto [sm_clock, mem_clock] = read_observed_clocks_mhz();
  env.observed_sm_clock_mhz = sm_clock;
  env.observed_mem_clock_mhz = mem_clock;

  env.os_note = "Linux (WSL2 Ubuntu) guest on Windows 11 host";
  env.timestamp_utc = capture_timestamp_utc();
  return env;
}

void finalize(BenchResult& result) {
  result.stats = compute_stats(result.raw_timings_ms);
  if (result.stats.median_ms > 0.0 && result.bytes_moved > 0.0) {
    // GB/s = bytes / 1e9 / (ms / 1000) = bytes / (ms * 1e6)
    result.effective_bandwidth_gb_s =
        result.bytes_moved / (result.stats.median_ms * 1.0e6);
  } else {
    result.effective_bandwidth_gb_s = 0.0;
  }
}

std::string to_json(const BenchResult& r) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"schema_version\":" << json_str(r.schema_version) << ",";
  oss << "\"kernel_family\":" << json_str(r.kernel_family) << ",";
  oss << "\"variant\":" << json_str(r.variant) << ",";
  oss << "\"description\":" << json_str(r.description) << ",";

  oss << "\"env\":{";
  oss << "\"gpu_name\":" << json_str(r.env.gpu_name) << ",";
  oss << "\"compute_capability\":" << json_str(std::to_string(r.env.compute_capability_major) +
                                                "." + std::to_string(r.env.compute_capability_minor))
      << ",";
  oss << "\"cuda_driver_version\":" << json_str(r.env.cuda_driver_version) << ",";
  oss << "\"cuda_runtime_version\":" << json_str(r.env.cuda_runtime_version) << ",";
  oss << "\"host_compiler_id\":" << json_str(r.env.host_compiler_id) << ",";
  oss << "\"host_compiler_version\":" << json_str(r.env.host_compiler_version) << ",";
  oss << "\"cuda_compiler_id\":" << json_str(r.env.cuda_compiler_id) << ",";
  oss << "\"cuda_compiler_version\":" << json_str(r.env.cuda_compiler_version) << ",";
  oss << "\"cuda_architectures\":" << json_str(r.env.cuda_architectures) << ",";
  oss << "\"build_type\":" << json_str(r.env.build_type) << ",";
  oss << "\"compile_flags_note\":" << json_str(r.env.compile_flags_note) << ",";
  oss << "\"locked_clocks\":" << json_bool(r.env.locked_clocks) << ",";
  oss << "\"clock_lock_note\":" << json_str(r.env.clock_lock_note) << ",";
  oss << "\"observed_sm_clock_mhz\":" << r.env.observed_sm_clock_mhz << ",";
  oss << "\"observed_mem_clock_mhz\":" << r.env.observed_mem_clock_mhz << ",";
  oss << "\"os_note\":" << json_str(r.env.os_note) << ",";
  oss << "\"timestamp_utc\":" << json_str(r.env.timestamp_utc);
  oss << "},";

  oss << "\"n\":" << r.n << ",";
  oss << "\"rows\":" << r.rows << ",";
  oss << "\"cols\":" << r.cols << ",";
  oss << "\"stride\":" << r.stride << ",";
  oss << "\"num_bins\":" << r.num_bins << ",";
  oss << "\"contention_profile\":" << json_str(r.contention_profile) << ",";
  oss << "\"dtype\":" << json_str(r.dtype) << ",";

  oss << "\"launch_config\":{";
  oss << "\"block\":[" << r.block_dim_x << "," << r.block_dim_y << "," << r.block_dim_z << "],";
  oss << "\"grid\":[" << r.grid_dim_x << "," << r.grid_dim_y << "," << r.grid_dim_z << "]";
  oss << "},";

  oss << "\"warmup_iters\":" << r.warmup_iters << ",";
  oss << "\"measured_reps\":" << r.measured_reps << ",";
  oss << "\"seed\":" << r.seed << ",";

  oss << "\"raw_timings_ms\":" << json_double_array(r.raw_timings_ms) << ",";
  oss << "\"stats_ms\":{";
  oss << "\"min\":" << r.stats.min_ms << ",";
  oss << "\"max\":" << r.stats.max_ms << ",";
  oss << "\"mean\":" << r.stats.mean_ms << ",";
  oss << "\"median\":" << r.stats.median_ms << ",";
  oss << "\"stddev\":" << r.stats.stddev_ms << ",";
  oss << "\"p25\":" << r.stats.p25_ms << ",";
  oss << "\"p75\":" << r.stats.p75_ms;
  oss << "},";

  oss << "\"bytes_moved\":" << r.bytes_moved << ",";
  oss << "\"effective_bandwidth_gb_s\":" << r.effective_bandwidth_gb_s << ",";
  oss << "\"gflops\":" << r.gflops << ",";

  oss << "\"tolerance_atol\":" << r.tolerance_atol << ",";
  oss << "\"tolerance_rtol\":" << r.tolerance_rtol << ",";
  oss << "\"correctness_passed\":" << json_bool(r.correctness_passed) << ",";
  oss << "\"correctness_note\":" << json_str(r.correctness_note) << ",";

  oss << "\"notes\":" << json_str(r.notes);
  oss << "}";
  return oss.str();
}

void append_jsonl(const std::string& path, const BenchResult& result) {
  ensure_parent_dir(path);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("kernelforge: failed to open '" + path + "' for append");
  }
  out << to_json(result) << "\n";
}

} // namespace kernelforge
