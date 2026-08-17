// Tolerance-based correctness comparison (FR6: "numerical tolerances by
// dtype"; FR4: "correctness tolerance" recorded per benchmark result).
#pragma once

#include <cstddef>
#include <string>

namespace kernelforge {

// Default tolerances for FP32 (ADR 0003: FP32-only MVP). Chosen to comfortably
// cover FP32 accumulation error on the sums this repo's kernels perform
// (single add/multiply-add per element, or an O(n) reduction for larger
// vector sizes) without being loose enough to mask a real correctness bug.
inline constexpr float kDefaultAtol = 1e-5f;
inline constexpr float kDefaultRtol = 1e-4f;

struct CompareResult {
  bool passed = true;
  std::size_t num_mismatches = 0;
  std::size_t first_mismatch_index = 0;
  float first_mismatch_actual = 0.0f;
  float first_mismatch_expected = 0.0f;
  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;

  // Human-readable one-line summary, suitable for a test failure message or
  // a BenchResult note.
  std::string summary() const;
};

// Elementwise |actual - expected| <= atol + rtol * |expected|, matching the
// common numpy.allclose convention. n == 0 trivially passes (FR6: "edge
// sizes (0/1 where legal)").
CompareResult allclose(const float* actual, const float* expected, std::size_t n,
                        float atol = kDefaultAtol, float rtol = kDefaultRtol);

} // namespace kernelforge
