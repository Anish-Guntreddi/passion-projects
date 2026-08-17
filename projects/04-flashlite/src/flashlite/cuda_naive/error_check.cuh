// CUDA error-check macros (mirrors KernelForge's src/common/error_check.cuh
// exactly, in the flashlite:: namespace). Every CUDA runtime API call and
// every kernel launch in this extension is wrapped by one of these. They
// throw flashlite::CudaError (a std::runtime_error) rather than
// abort()/exit(): pybind11 translates any uncaught std::runtime_error
// escaping a bound function into a Python RuntimeError automatically, so
// a bad launch config or an OOM surfaces as a clear, catchable Python
// exception instead of crashing the interpreter (spec hard constraint:
// "unsupported shapes must fail clearly").
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace flashlite {

class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const char* expr, const char* file, int line)
      : std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": CUDA error in `" +
                            expr + "`: " + cudaGetErrorString(code)),
        code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

}  // namespace flashlite

// Checks the cudaError_t returned by `expr`. Use for every runtime API call
// (cudaMalloc, cudaEventRecord, cudaMemcpy, ...).
#define FL_CUDA_CHECK(expr)                                                                \
  do {                                                                                     \
    const cudaError_t fl_cuda_check_err__ = (expr);                                        \
    if (fl_cuda_check_err__ != cudaSuccess) {                                              \
      throw ::flashlite::CudaError(fl_cuda_check_err__, #expr, __FILE__, __LINE__);        \
    }                                                                                       \
  } while (0)

// Checks for asynchronous kernel-launch errors. Call immediately after every
// `kernel<<<grid, block>>>(...)` launch -- this catches a bad launch config
// (too many threads, too much shared memory per block) immediately rather
// than at the next unrelated CUDA call.
#define FL_CUDA_CHECK_LAST_ERROR() FL_CUDA_CHECK(cudaGetLastError())
