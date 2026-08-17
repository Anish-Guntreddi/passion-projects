// Host wall-clock timer, used for CPU reference-implementation timing and
// for host-side end-to-end (including transfer) timing when explicitly
// needed. Not used for kernel-only timing — see gpu_timer.cuh for that.
#pragma once

#include <chrono>

namespace kernelforge {

class CpuTimer {
public:
  void start() { start_ = clock::now(); }

  double stop_ms() {
    const auto end = clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_{};
};

} // namespace kernelforge
