#include "common/reference_ops.hpp"

#include <cmath>
#include <limits>
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

void gemm(const float* a, const float* b, float* c, int m, int n, int k) {
  for (int mi = 0; mi < m; ++mi) {
    for (int ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (int ki = 0; ki < k; ++ki) {
        acc += static_cast<double>(a[static_cast<std::size_t>(mi) * k + ki]) *
               static_cast<double>(b[static_cast<std::size_t>(ki) * n + ni]);
      }
      c[static_cast<std::size_t>(mi) * n + ni] = static_cast<float>(acc);
    }
  }
}

void softmax_rows(const float* in, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row_in = in + static_cast<std::size_t>(r) * cols;
    float* row_out = out + static_cast<std::size_t>(r) * cols;

    float row_max = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cols; ++c) {
      if (row_in[c] > row_max) row_max = row_in[c];
    }

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += std::exp(static_cast<double>(row_in[c]) - static_cast<double>(row_max));
    }

    for (int c = 0; c < cols; ++c) {
      row_out[c] = static_cast<float>(
          std::exp(static_cast<double>(row_in[c]) - static_cast<double>(row_max)) / sum);
    }
  }
}

void rmsnorm_rows(const float* in, const float* gamma, float* out, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* row_in = in + static_cast<std::size_t>(r) * cols;
    float* row_out = out + static_cast<std::size_t>(r) * cols;

    double sum_sq = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double v = static_cast<double>(row_in[c]);
      sum_sq += v * v;
    }
    const double mean_sq = (cols > 0) ? (sum_sq / static_cast<double>(cols)) : 0.0;
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

    for (int c = 0; c < cols; ++c) {
      row_out[c] = static_cast<float>(static_cast<double>(row_in[c]) * inv_rms) * gamma[c];
    }
  }
}

} // namespace kernelforge::reference
