#include "common/device_query.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

#include "common/error_check.cuh"

namespace kernelforge {

DeviceInfo query_device_info(int device_id) {
  DeviceInfo info;
  info.device_id = device_id;

  int device_count = 0;
  KF_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("kernelforge: no CUDA-capable device found");
  }
  if (device_id < 0 || device_id >= device_count) {
    throw std::runtime_error("kernelforge: requested device_id out of range");
  }

  cudaDeviceProp prop{};
  KF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  info.name = prop.name;
  info.compute_capability_major = prop.major;
  info.compute_capability_minor = prop.minor;
  info.total_global_mem_bytes = prop.totalGlobalMem;
  info.multiprocessor_count = prop.multiProcessorCount;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
  info.warp_size = prop.warpSize;
  info.shared_mem_per_block_bytes = prop.sharedMemPerBlock;
  info.shared_mem_per_multiprocessor_bytes = prop.sharedMemPerMultiprocessor;
  info.regs_per_block = prop.regsPerBlock;
  info.clock_rate_khz = prop.clockRate;
  info.memory_clock_rate_khz = prop.memoryClockRate;
  info.memory_bus_width_bits = prop.memoryBusWidth;
  info.l2_cache_size_bytes = static_cast<std::size_t>(prop.l2CacheSize);
  info.ecc_enabled = prop.ECCEnabled;
  info.concurrent_kernels = prop.concurrentKernels;
  info.async_engine_count = prop.asyncEngineCount;
  info.unified_addressing = prop.unifiedAddressing;

  KF_CUDA_CHECK(cudaDriverGetVersion(&info.cuda_driver_version));
  KF_CUDA_CHECK(cudaRuntimeGetVersion(&info.cuda_runtime_version));

  info.theoretical_peak_bandwidth_gb_s =
      2.0 * static_cast<double>(info.memory_clock_rate_khz) * 1e3 *
      (static_cast<double>(info.memory_bus_width_bits) / 8.0) / 1e9;

  return info;
}

std::string format_cuda_version(int encoded_version) {
  const int major = encoded_version / 1000;
  const int minor = (encoded_version % 1000) / 10;
  std::ostringstream oss;
  oss << major << "." << minor;
  return oss.str();
}

bool meets_min_capability(const DeviceInfo& info, int min_major, int min_minor) {
  if (info.compute_capability_major != min_major) {
    return info.compute_capability_major > min_major;
  }
  return info.compute_capability_minor >= min_minor;
}

void require_capability_or_throw(const DeviceInfo& info, int min_major, int min_minor) {
  if (!meets_min_capability(info, min_major, min_minor)) {
    std::ostringstream oss;
    oss << "kernelforge: device '" << info.name
        << "' has compute capability " << info.compute_capability_major << "."
        << info.compute_capability_minor << ", but this build requires >= "
        << min_major << "." << min_minor
        << ". Refusing to run rather than risk silently wrong behavior "
           "(see docs/decisions/0002-minimum-compute-capability.md).";
    throw std::runtime_error(oss.str());
  }
}

std::string to_human_readable(const DeviceInfo& info) {
  std::ostringstream oss;
  oss << "KernelForge device-info\n"
      << "========================\n"
      << "Device index         : " << info.device_id << "\n"
      << "Name                 : " << info.name << "\n"
      << "Compute capability   : " << info.compute_capability_major << "."
      << info.compute_capability_minor << "\n"
      << "Total global memory  : "
      << (static_cast<double>(info.total_global_mem_bytes) / (1024.0 * 1024.0 * 1024.0))
      << " GiB\n"
      << "SM count             : " << info.multiprocessor_count << "\n"
      << "Warp size            : " << info.warp_size << "\n"
      << "Max threads/block    : " << info.max_threads_per_block << "\n"
      << "Max threads/SM       : " << info.max_threads_per_multiprocessor << "\n"
      << "Shared mem/block     : " << info.shared_mem_per_block_bytes << " bytes\n"
      << "Shared mem/SM        : " << info.shared_mem_per_multiprocessor_bytes << " bytes\n"
      << "Registers/block      : " << info.regs_per_block << "\n"
      << "Core clock           : " << (info.clock_rate_khz / 1000.0) << " MHz\n"
      << "Memory clock         : " << (info.memory_clock_rate_khz / 1000.0) << " MHz\n"
      << "Memory bus width     : " << info.memory_bus_width_bits << " bits\n"
      << "L2 cache size        : " << (static_cast<double>(info.l2_cache_size_bytes) / 1024.0)
      << " KiB\n"
      << "ECC enabled          : " << (info.ecc_enabled ? "yes" : "no") << "\n"
      << "Concurrent kernels   : " << (info.concurrent_kernels ? "yes" : "no") << "\n"
      << "Async engine count   : " << info.async_engine_count << "\n"
      << "Unified addressing   : " << (info.unified_addressing ? "yes" : "no") << "\n"
      << "CUDA driver version  : " << format_cuda_version(info.cuda_driver_version) << "\n"
      << "CUDA runtime version : " << format_cuda_version(info.cuda_runtime_version) << "\n"
      << "Theoretical peak BW  : " << info.theoretical_peak_bandwidth_gb_s << " GB/s\n";
  return oss.str();
}

std::string to_json(const DeviceInfo& info) {
  std::ostringstream oss;
  oss << "{"
      << "\"device_id\":" << info.device_id << ","
      << "\"name\":\"" << info.name << "\","
      << "\"compute_capability_major\":" << info.compute_capability_major << ","
      << "\"compute_capability_minor\":" << info.compute_capability_minor << ","
      << "\"total_global_mem_bytes\":" << info.total_global_mem_bytes << ","
      << "\"multiprocessor_count\":" << info.multiprocessor_count << ","
      << "\"max_threads_per_block\":" << info.max_threads_per_block << ","
      << "\"max_threads_per_multiprocessor\":" << info.max_threads_per_multiprocessor << ","
      << "\"warp_size\":" << info.warp_size << ","
      << "\"shared_mem_per_block_bytes\":" << info.shared_mem_per_block_bytes << ","
      << "\"shared_mem_per_multiprocessor_bytes\":"
      << info.shared_mem_per_multiprocessor_bytes << ","
      << "\"regs_per_block\":" << info.regs_per_block << ","
      << "\"clock_rate_khz\":" << info.clock_rate_khz << ","
      << "\"memory_clock_rate_khz\":" << info.memory_clock_rate_khz << ","
      << "\"memory_bus_width_bits\":" << info.memory_bus_width_bits << ","
      << "\"l2_cache_size_bytes\":" << info.l2_cache_size_bytes << ","
      << "\"ecc_enabled\":" << (info.ecc_enabled ? "true" : "false") << ","
      << "\"concurrent_kernels\":" << (info.concurrent_kernels ? "true" : "false") << ","
      << "\"async_engine_count\":" << info.async_engine_count << ","
      << "\"unified_addressing\":" << (info.unified_addressing ? "true" : "false") << ","
      << "\"cuda_driver_version\":\"" << format_cuda_version(info.cuda_driver_version) << "\","
      << "\"cuda_runtime_version\":\"" << format_cuda_version(info.cuda_runtime_version) << "\","
      << "\"theoretical_peak_bandwidth_gb_s\":" << info.theoretical_peak_bandwidth_gb_s << "}";
  return oss.str();
}

} // namespace kernelforge
