// Device-info query (FR3: "device-info command") and capability-check
// helpers (hard constraint 8: "fail loudly on unsupported hardware
// features"; ADR 0001/0002).
#pragma once

#include <cstddef>
#include <string>

namespace kernelforge {

struct DeviceInfo {
  int device_id = 0;
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_global_mem_bytes = 0;
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  int max_threads_per_multiprocessor = 0;
  int warp_size = 0;
  std::size_t shared_mem_per_block_bytes = 0;
  std::size_t shared_mem_per_multiprocessor_bytes = 0;
  int regs_per_block = 0;
  int clock_rate_khz = 0;
  int memory_clock_rate_khz = 0;
  int memory_bus_width_bits = 0;
  std::size_t l2_cache_size_bytes = 0;
  int ecc_enabled = 0;
  int concurrent_kernels = 0;
  int async_engine_count = 0;
  int unified_addressing = 0;

  int cuda_driver_version = 0;  // cudaDriverGetVersion(), e.g. 13010
  int cuda_runtime_version = 0; // cudaRuntimeGetVersion(), e.g. 12060

  // Derived: 2 (DDR) * memory_clock_rate_khz * 1e3 * (bus_width_bits / 8) / 1e9
  double theoretical_peak_bandwidth_gb_s = 0.0;
};

DeviceInfo query_device_info(int device_id = 0);

// "major.minor" from a CUDA-encoded version int (e.g. 12060 -> "12.6").
std::string format_cuda_version(int encoded_version);

std::string to_human_readable(const DeviceInfo& info);
std::string to_json(const DeviceInfo& info);

bool meets_min_capability(const DeviceInfo& info, int min_major, int min_minor);

// Throws std::runtime_error if `info` does not meet the minimum capability.
void require_capability_or_throw(const DeviceInfo& info, int min_major, int min_minor);

} // namespace kernelforge
