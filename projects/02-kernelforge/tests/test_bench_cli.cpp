// Unit tests for apps/bench_cli.hpp's validate_methodology() -- the
// ADR 0005 methodology floor (>= 20 measured reps, warmup >= 0) enforced
// at the CLI level in every bench_*_main.cpp, before any GPU work runs.
// Pure host-side logic, no GPU launch involved.
#include "bench_cli.hpp"
#include "common/testing.hpp"

KF_TEST(BenchCli, DefaultArgsPassMethodologyValidation) {
  kernelforge::apps::CliArgs args;
  KF_EXPECT_TRUE(!kernelforge::apps::validate_methodology(args).has_value());
}

KF_TEST(BenchCli, RepsBelowMinimumRejected) {
  kernelforge::apps::CliArgs args;
  args.measured_reps = kernelforge::apps::kMinMeasuredReps - 1;
  KF_EXPECT_TRUE(kernelforge::apps::validate_methodology(args).has_value());

  args.measured_reps = 1;
  KF_EXPECT_TRUE(kernelforge::apps::validate_methodology(args).has_value());

  args.measured_reps = 0;
  KF_EXPECT_TRUE(kernelforge::apps::validate_methodology(args).has_value());
}

KF_TEST(BenchCli, RepsAtOrAboveMinimumAccepted) {
  kernelforge::apps::CliArgs args;
  args.measured_reps = kernelforge::apps::kMinMeasuredReps;
  KF_EXPECT_TRUE(!kernelforge::apps::validate_methodology(args).has_value());

  args.measured_reps = 1000;
  KF_EXPECT_TRUE(!kernelforge::apps::validate_methodology(args).has_value());
}

KF_TEST(BenchCli, NegativeWarmupRejected) {
  kernelforge::apps::CliArgs args;
  args.warmup_iters = -1;
  KF_EXPECT_TRUE(kernelforge::apps::validate_methodology(args).has_value());
}

KF_TEST(BenchCli, ZeroWarmupAccepted) {
  kernelforge::apps::CliArgs args;
  args.warmup_iters = 0;
  KF_EXPECT_TRUE(!kernelforge::apps::validate_methodology(args).has_value());
}
