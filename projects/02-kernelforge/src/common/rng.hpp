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

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
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

// Phase 3 (histogram/atomics lab): which input-value distribution to
// generate. See make_histogram_input below for exactly what each one does
// to global-atomic contention, and src/kernels/histogram/README.md for the
// measured before/after.
enum class ContentionProfile { kUniform, kSkewed };

inline const char* to_string(ContentionProfile profile) {
  return profile == ContentionProfile::kSkewed ? "skewed" : "uniform";
}

inline ContentionProfile contention_profile_from_string(const std::string& s) {
  if (s == "skewed") return ContentionProfile::kSkewed;
  if (s == "uniform") return ContentionProfile::kUniform;
  throw std::invalid_argument("contention_profile_from_string: expected 'uniform' or 'skewed', got '" +
                               s + "'");
}

// Generates n values in [0, 1) intended to be bucketed by
// bin = floor(value * num_bins) (see kernels/histogram/histogram_common.cuh).
//
// kUniform ("low contention"): value ~ Uniform(0, 1) directly. After
// binning, load spreads roughly evenly across all num_bins bins -- a given
// bin's global-memory counter is touched by only ~n/num_bins threads out
// of the whole launch, so within any single warp's 32 concurrent
// atomicAdd requests, two threads landing on the exact same bin is
// comparatively rare (birthday-paradox-rare, not zero, but low).
//
// kSkewed ("high contention"): value = uniform^kSkewExponent. Raising a
// value in [0, 1) to a large power concentrates the resulting
// distribution's mass near 0 (a standard power-law-style skew, similar in
// shape to real skewed categorical data, e.g. Zipfian word frequencies).
// After binning, the large majority of the n elements land in the first
// handful of bins out of num_bins -- many concurrent threads' atomicAdd
// requests now target the SAME few global-memory addresses. This is
// exactly the pathological case shared-memory privatization
// (kernels/histogram/histogram_privatized.cuh) exists to fix; see
// kernels/histogram/README.md for the measured low- vs high-contention
// comparison across all three histogram kernel variants.
inline std::vector<float> make_histogram_input(std::size_t n, std::uint64_t seed,
                                                ContentionProfile profile) {
  constexpr float kSkewExponent = 24.0f;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> values(n);
  for (auto& v : values) {
    const float u = dist(rng);
    v = (profile == ContentionProfile::kSkewed) ? std::pow(u, kSkewExponent) : u;
  }
  return values;
}

} // namespace kernelforge
