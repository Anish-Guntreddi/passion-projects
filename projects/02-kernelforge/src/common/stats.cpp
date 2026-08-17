#include "common/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace kernelforge {

namespace {
// Linear-interpolation percentile (numpy's default "linear" method), on an
// already-sorted vector. p in [0, 1].
double percentile(const std::vector<double>& sorted, double p) {
  const std::size_t n = sorted.size();
  if (n == 1) return sorted[0];
  const double idx = p * static_cast<double>(n - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  if (lo == hi) return sorted[lo];
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}
} // namespace

TimingStats compute_stats(std::vector<double> samples_ms) {
  if (samples_ms.empty()) {
    throw std::invalid_argument("compute_stats: samples_ms must be non-empty");
  }
  std::sort(samples_ms.begin(), samples_ms.end());

  TimingStats stats;
  stats.min_ms = samples_ms.front();
  stats.max_ms = samples_ms.back();

  double sum = 0.0;
  for (double v : samples_ms) sum += v;
  stats.mean_ms = sum / static_cast<double>(samples_ms.size());

  double sq_diff_sum = 0.0;
  for (double v : samples_ms) {
    const double d = v - stats.mean_ms;
    sq_diff_sum += d * d;
  }
  stats.stddev_ms = samples_ms.size() > 1
                         ? std::sqrt(sq_diff_sum / static_cast<double>(samples_ms.size() - 1))
                         : 0.0;

  stats.median_ms = percentile(samples_ms, 0.5);
  stats.p25_ms = percentile(samples_ms, 0.25);
  stats.p75_ms = percentile(samples_ms, 0.75);
  return stats;
}

} // namespace kernelforge
