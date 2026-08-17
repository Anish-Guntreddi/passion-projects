#include "common/reference_ops.hpp"

#include <stdexcept>
#include <string>

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

float reduce_sum(const float* in, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(in[i]);
  }
  return static_cast<float>(acc);
}

void inclusive_scan(const float* in, float* out, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(in[i]);
    out[i] = static_cast<float>(acc);
  }
}

void histogram(const float* in, std::size_t n, int* hist_out, int num_bins) {
  // See the .hpp doc comment: without this, num_bins <= 0 sends the clamp
  // formula below to bin = -1 and writes out of bounds.
  if (num_bins <= 0) {
    throw std::invalid_argument("kernelforge::reference::histogram: num_bins (" +
                                 std::to_string(num_bins) + ") must be > 0");
  }
  for (int b = 0; b < num_bins; ++b) {
    hist_out[b] = 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    int bin = static_cast<int>(in[i] * static_cast<float>(num_bins));
    if (bin < 0) bin = 0;
    if (bin >= num_bins) bin = num_bins - 1;
    hist_out[bin] += 1;
  }
}

} // namespace kernelforge::reference
