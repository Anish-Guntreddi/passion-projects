#include "common/error_check.cuh"

#include <sstream>

namespace kernelforge {

namespace {
std::string build_message(cudaError_t code, const char* expr, const char* file,
                           int line) {
  std::ostringstream oss;
  oss << file << ":" << line << ": CUDA error " << static_cast<int>(code)
      << " (" << cudaGetErrorName(code) << "): " << cudaGetErrorString(code)
      << "\n  in expression: " << expr;
  return oss.str();
}
} // namespace

CudaError::CudaError(cudaError_t code, const char* expr, const char* file,
                      int line)
    : std::runtime_error(build_message(code, expr, file, line)), code_(code) {}

} // namespace kernelforge
