// Tests for the benchmark result schema itself (common/stats.hpp,
// common/bench_result.hpp) -- ADR 0010. These are pure host-side logic
// tests (no GPU kernel launch), verifying the statistics math and JSON
// serialization independently of any specific kernel's numbers.
#include <string>
#include <vector>

#include "common/bench_result.hpp"
#include "common/stats.hpp"
#include "common/testing.hpp"

KF_TEST(Stats, MedianOfOddCountIsMiddleElement) {
  const auto stats = kernelforge::compute_stats({5.0, 1.0, 3.0, 2.0, 4.0});
  KF_EXPECT_NEAR(stats.median_ms, 3.0, 1e-9);
  KF_EXPECT_NEAR(stats.min_ms, 1.0, 1e-9);
  KF_EXPECT_NEAR(stats.max_ms, 5.0, 1e-9);
  KF_EXPECT_NEAR(stats.mean_ms, 3.0, 1e-9);
}

KF_TEST(Stats, SingleSampleIsDegenerateButValid) {
  const auto stats = kernelforge::compute_stats({7.5});
  KF_EXPECT_NEAR(stats.median_ms, 7.5, 1e-9);
  KF_EXPECT_NEAR(stats.min_ms, 7.5, 1e-9);
  KF_EXPECT_NEAR(stats.max_ms, 7.5, 1e-9);
  KF_EXPECT_NEAR(stats.stddev_ms, 0.0, 1e-9);
}

KF_TEST(Stats, IqrBoundsAreOrderedAroundMedian) {
  std::vector<double> samples;
  for (int i = 1; i <= 100; ++i) samples.push_back(static_cast<double>(i));
  const auto stats = kernelforge::compute_stats(samples);
  KF_EXPECT_TRUE(stats.p25_ms < stats.median_ms);
  KF_EXPECT_TRUE(stats.median_ms < stats.p75_ms);
  KF_EXPECT_NEAR(stats.p25_ms, 25.75, 0.5);
  KF_EXPECT_NEAR(stats.p75_ms, 75.25, 0.5);
}

KF_TEST(BenchResult, FinalizeComputesBandwidthFromMedianAndBytes) {
  kernelforge::BenchResult r;
  r.raw_timings_ms = {1.0, 1.0, 1.0}; // median = 1.0 ms
  r.bytes_moved = 1.0e6;              // 1,000,000 bytes in 1 ms -> 1 GB/s
  kernelforge::finalize(r);
  KF_EXPECT_NEAR(r.effective_bandwidth_gb_s, 1.0, 1e-6);
}

KF_TEST(BenchResult, FinalizeHandlesZeroBytesMovedGracefully) {
  kernelforge::BenchResult r;
  r.raw_timings_ms = {1.0, 2.0, 3.0};
  r.bytes_moved = 0.0;
  kernelforge::finalize(r);
  KF_EXPECT_NEAR(r.effective_bandwidth_gb_s, 0.0, 1e-9);
}

KF_TEST(BenchResult, ToJsonProducesWellFormedBracesAndRequiredFields) {
  kernelforge::BenchResult r;
  r.kernel_family = "vector_add";
  r.variant = "v1_naive";
  r.n = 1024;
  r.raw_timings_ms = {0.1, 0.2, 0.15};
  r.bytes_moved = 1024.0 * 4 * 3;
  r.correctness_passed = true;
  r.correctness_note = "allclose PASSED";
  kernelforge::finalize(r);

  const std::string json = kernelforge::to_json(r);
  KF_EXPECT_TRUE(!json.empty());
  KF_EXPECT_TRUE(json.front() == '{');
  KF_EXPECT_TRUE(json.back() == '}');
  KF_EXPECT_TRUE(json.find("\"kernel_family\":\"vector_add\"") != std::string::npos);
  KF_EXPECT_TRUE(json.find("\"schema_version\":\"1.0\"") != std::string::npos);
  KF_EXPECT_TRUE(json.find("\"raw_timings_ms\":[0.1,0.2,0.15]") != std::string::npos);
  KF_EXPECT_TRUE(json.find("\"correctness_passed\":true") != std::string::npos);
  // Balanced braces: a cheap sanity check that nothing left a dangling quote.
  int depth = 0;
  bool balanced = true;
  for (char c : json) {
    if (c == '{') ++depth;
    if (c == '}') --depth;
    if (depth < 0) balanced = false;
  }
  KF_EXPECT_TRUE(balanced && depth == 0);
}

KF_TEST(BenchResult, JsonEscapesEmbeddedQuotesAndBackslashes) {
  kernelforge::BenchResult r;
  r.kernel_family = "vector_add";
  r.notes = "contains \"quotes\" and a backslash \\ here";
  r.raw_timings_ms = {1.0};
  kernelforge::finalize(r);
  const std::string json = kernelforge::to_json(r);
  KF_EXPECT_TRUE(json.find("\\\"quotes\\\"") != std::string::npos);
  KF_EXPECT_TRUE(json.find("\\\\") != std::string::npos);
}
