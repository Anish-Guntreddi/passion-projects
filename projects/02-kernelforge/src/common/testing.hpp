// Minimal self-registering test framework used by every tests/*.cpp file
// (ADR 0009: lightweight framework instead of GoogleTest).
//
// Usage:
//   #include "common/testing.hpp"
//   KF_TEST(VectorAdd, MatchesReferenceOnRandomInput) {
//     KF_EXPECT_TRUE(...);
//     KF_ASSERT_EQ(a, b);   // fatal: aborts just this test case
//   }
//   // in exactly one .cpp (tests/test_main.cpp):
//   int main(int argc, char** argv) { return kernelforge::testing::run_all(argc, argv); }
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace kernelforge::testing {

struct TestCase {
  std::string suite;
  std::string name;
  std::function<void()> fn;
};

// Thrown by KF_ASSERT_* to unwind out of the current test body after a
// fatal check fails. Caught by run_all() so one bad assertion fails only
// its own test case, not the whole binary.
struct FatalTestFailure {};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> cases;
  return cases;
}

struct Registrar {
  Registrar(std::string suite, std::string name, std::function<void()> fn) {
    registry().push_back(TestCase{std::move(suite), std::move(name), std::move(fn)});
  }
};

// Failures recorded by the currently-executing test case. Cleared by
// run_all() before each test case runs.
inline std::vector<std::string>& current_failures() {
  static std::vector<std::string> failures;
  return failures;
}

inline void record_failure(const std::string& message) {
  current_failures().push_back(message);
  std::fprintf(stderr, "    FAILED: %s\n", message.c_str());
}

// Runs every registered test case (optionally filtered by
// `--filter=substring` matched against "Suite.Name"), prints a GoogleTest-
// style summary, and returns 0 iff every test passed.
int run_all(int argc, char** argv);

} // namespace kernelforge::testing

#define KF_TEST(suite, name)                                                          \
  static void kf_test_body_##suite##_##name();                                        \
  namespace {                                                                          \
  ::kernelforge::testing::Registrar kf_test_reg_##suite##_##name(                      \
      #suite, #name, kf_test_body_##suite##_##name);                                   \
  }                                                                                     \
  static void kf_test_body_##suite##_##name()

#define KF_EXPECT_TRUE(cond)                                                          \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::ostringstream kf_oss__;                                                     \
      kf_oss__ << __FILE__ << ":" << __LINE__ << ": EXPECT_TRUE(" #cond ") failed";    \
      ::kernelforge::testing::record_failure(kf_oss__.str());                          \
    }                                                                                  \
  } while (0)

#define KF_ASSERT_TRUE(cond)                                                          \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::ostringstream kf_oss__;                                                     \
      kf_oss__ << __FILE__ << ":" << __LINE__ << ": ASSERT_TRUE(" #cond ") failed";    \
      ::kernelforge::testing::record_failure(kf_oss__.str());                          \
      throw ::kernelforge::testing::FatalTestFailure{};                                \
    }                                                                                  \
  } while (0)

#define KF_EXPECT_EQ(a, b)                                                            \
  do {                                                                                 \
    const auto kf_a__ = (a);                                                           \
    const auto kf_b__ = (b);                                                           \
    if (!(kf_a__ == kf_b__)) {                                                         \
      std::ostringstream kf_oss__;                                                     \
      kf_oss__ << __FILE__ << ":" << __LINE__ << ": EXPECT_EQ(" #a ", " #b ") failed: " \
                << kf_a__ << " != " << kf_b__;                                         \
      ::kernelforge::testing::record_failure(kf_oss__.str());                          \
    }                                                                                  \
  } while (0)

#define KF_EXPECT_NEAR(a, b, tol)                                                     \
  do {                                                                                 \
    const double kf_a__ = static_cast<double>(a);                                      \
    const double kf_b__ = static_cast<double>(b);                                      \
    const double kf_diff__ = std::fabs(kf_a__ - kf_b__);                                \
    if (kf_diff__ > static_cast<double>(tol)) {                                         \
      std::ostringstream kf_oss__;                                                     \
      kf_oss__ << __FILE__ << ":" << __LINE__ << ": EXPECT_NEAR(" #a ", " #b ") failed: " \
                << kf_a__ << " vs " << kf_b__ << " (diff=" << kf_diff__                \
                << " > tol=" << tol << ")";                                            \
      ::kernelforge::testing::record_failure(kf_oss__.str());                          \
    }                                                                                  \
  } while (0)
