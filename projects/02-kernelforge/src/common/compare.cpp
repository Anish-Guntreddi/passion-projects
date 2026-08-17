#include "common/compare.hpp"

#include <cmath>
#include <sstream>

namespace kernelforge {

std::string CompareResult::summary() const {
  std::ostringstream oss;
  if (passed) {
    oss << "allclose PASSED (max_abs_diff=" << max_abs_diff
        << ", max_rel_diff=" << max_rel_diff << ")";
  } else {
    oss << "allclose FAILED: " << num_mismatches
        << " mismatch(es); first at index " << first_mismatch_index
        << " actual=" << first_mismatch_actual
        << " expected=" << first_mismatch_expected
        << "; max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff;
  }
  return oss.str();
}

CompareResult allclose(const float* actual, const float* expected, std::size_t n,
                        float atol, float rtol) {
  CompareResult result;
  for (std::size_t i = 0; i < n; ++i) {
    const float a = actual[i];
    const float e = expected[i];
    const float abs_diff = std::fabs(a - e);
    const float rel_diff = abs_diff / (std::fabs(e) + 1e-30f);

    if (abs_diff > result.max_abs_diff) result.max_abs_diff = abs_diff;
    if (rel_diff > result.max_rel_diff) result.max_rel_diff = rel_diff;

    const bool ok = abs_diff <= (atol + rtol * std::fabs(e));
    if (!ok) {
      if (result.num_mismatches == 0) {
        result.first_mismatch_index = i;
        result.first_mismatch_actual = a;
        result.first_mismatch_expected = e;
      }
      ++result.num_mismatches;
      result.passed = false;
    }
  }
  return result;
}

} // namespace kernelforge
