#include "common/reference_ops.hpp"

namespace kernelforge::reference {

void vector_add(const float* x, const float* y, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = x[i] + y[i];
  }
}

void saxpy(float a, const float* x, const float* y_in, float* y_out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    y_out[i] = a * x[i] + y_in[i];
  }
}

void transpose(const float* in, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows) +
          static_cast<std::size_t>(r)] =
          in[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
             static_cast<std::size_t>(c)];
    }
  }
}

} // namespace kernelforge::reference
