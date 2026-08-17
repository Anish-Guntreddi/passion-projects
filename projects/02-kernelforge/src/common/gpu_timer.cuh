// CUDA-event-based GPU timer (FR3: "timers (CUDA events)").
//
// Measures device-side elapsed time between start() and stop_ms() using
// cudaEvent_t, which is the correct way to time kernel execution (as
// opposed to wall-clock host timers, which include host-side launch
// overhead and any host work that races ahead of the async kernel launch).
//
// This measures *kernel* time only. Transfer time (H2D/D2H) is measured
// separately where relevant, per FR4: "Kernel time and transfer time are
// never mixed unless explicitly measuring end-to-end."
#pragma once

#include <cuda_runtime.h>

#include "common/error_check.cuh"

namespace kernelforge {

class GpuTimer {
public:
  GpuTimer() {
    KF_CUDA_CHECK(cudaEventCreate(&start_));
    KF_CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = 0) { KF_CUDA_CHECK(cudaEventRecord(start_, stream)); }

  // Records the stop event, synchronizes on it, and returns elapsed
  // milliseconds since start(). Blocks the host thread until the recorded
  // work completes (this is intentional and required for accurate timing).
  float stop_ms(cudaStream_t stream = 0) {
    KF_CUDA_CHECK(cudaEventRecord(stop_, stream));
    KF_CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed_ms = 0.0f;
    KF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return elapsed_ms;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

} // namespace kernelforge
