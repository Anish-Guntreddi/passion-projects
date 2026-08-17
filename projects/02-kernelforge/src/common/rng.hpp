// Deterministic seeded RNG helpers for generating reference/test input data
// (FR6: "deterministic seeds").
//
// std::mt19937_64 is used (not std::random_device) specifically because it
// is deterministic given a seed: the same seed always produces the same
// input data, on any machine, forever. This is required so that a test
// failure is reproducible and so benchmark inputs are identical across
// variants being compared (ADR "one optimization variable per experiment"
// — the *input data* must not be a hidden second variable).
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace kernelforge {

// Fixed default seed used across this repo's tests and benchmarks unless a
// specific test intentionally sweeps seeds. Recorded verbatim in every
// committed BenchResult (see bench_result.hpp) so results are reproducible.
inline constexpr std::uint64_t kDefaultSeed = 123456789ULL;

inline std::vector<float> make_random_vector(std::size_t n, std::uint64_t seed,
                                               float lo = -1.0f, float hi = 1.0f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> values(n);
  for (auto& v : values) {
    v = dist(rng);
  }
  return values;
}

} // namespace kernelforge
