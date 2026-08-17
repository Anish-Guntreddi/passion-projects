// CUDA error-check macros (FR3: "CUDA error macros").
//
// Every CUDA runtime API call and every kernel launch in this repo is
// wrapped by one of these. They throw kernelforge::CudaError rather than
// abort()/exit() so that:
//   - test binaries can catch the exception and report a clean failure
//     instead of crashing the whole test process (one bad size shouldn't
//     kill every other test case), and
//   - benchmark binaries can let it propagate to main() and exit non-zero
//     with a clear message (hard constraint 8: fail loudly).
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kernelforge {

class CudaError : public std::runtime_error {
public:
  CudaError(cudaError_t code, const char* expr, const char* file, int line);

  cudaError_t code() const noexcept { return code_; }

private:
  cudaError_t code_;
};

} // namespace kernelforge

// Checks the cudaError_t returned by `expr`. Use for every runtime API call
// (cudaMalloc, cudaMemcpy, cudaEventRecord, cudaDeviceSynchronize, ...).
#define KF_CUDA_CHECK(expr)                                                  \
  do {                                                                       \
    const cudaError_t kf_cuda_check_err__ = (expr);                         \
    if (kf_cuda_check_err__ != cudaSuccess) {                                \
      throw ::kernelforge::CudaError(kf_cuda_check_err__, #expr, __FILE__,   \
                                      __LINE__);                             \
    }                                                                        \
  } while (0)

// Checks for asynchronous kernel-launch errors. Call immediately after every
// `kernel<<<grid, block>>>(...)` launch, before assuming the launch queued
// successfully (this catches bad launch configs, e.g. too many threads or
// too much shared memory per block, immediately rather than at the next
// unrelated CUDA call).
#define KF_CUDA_CHECK_LAST_ERROR() KF_CUDA_CHECK(cudaGetLastError())
