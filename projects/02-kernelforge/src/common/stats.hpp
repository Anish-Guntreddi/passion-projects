// Distribution statistics over a set of raw timing samples (ADR 0005:
// "median + IQR", FR4: "median + distribution (never a single timing)").
#pragma once

#include <vector>

namespace kernelforge {

struct TimingStats {
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double stddev_ms = 0.0;
  double p25_ms = 0.0; // IQR lower bound
  double p75_ms = 0.0; // IQR upper bound
};

// Computes summary statistics over `samples_ms`. Does not mutate the input
// (takes a copy internally to sort). Requires samples_ms non-empty.
TimingStats compute_stats(std::vector<double> samples_ms);

} // namespace kernelforge
