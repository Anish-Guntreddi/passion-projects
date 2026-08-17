#include <exception>
#include <string>

#include "common/device_query.hpp"
#include "common/testing.hpp"

using kernelforge::DeviceInfo;

KF_TEST(DeviceInfo, QueryDoesNotThrowAndReportsSm89) {
  const DeviceInfo info = kernelforge::query_device_info(0);
  KF_EXPECT_TRUE(!info.name.empty());
  // ADR 0001: this repo is pinned to the developer's RTX 4090, sm_89.
  KF_EXPECT_EQ(info.compute_capability_major, 8);
  KF_EXPECT_EQ(info.compute_capability_minor, 9);
  KF_EXPECT_TRUE(info.multiprocessor_count > 0);
  KF_EXPECT_TRUE(info.warp_size == 32);
  KF_EXPECT_TRUE(info.total_global_mem_bytes > 0);
  KF_EXPECT_TRUE(info.theoretical_peak_bandwidth_gb_s > 0.0);
}

KF_TEST(DeviceInfo, MeetsMinCapabilityLogic) {
  DeviceInfo info;
  info.compute_capability_major = 8;
  info.compute_capability_minor = 9;

  KF_EXPECT_TRUE(kernelforge::meets_min_capability(info, 8, 9));
  KF_EXPECT_TRUE(kernelforge::meets_min_capability(info, 8, 0));
  KF_EXPECT_TRUE(kernelforge::meets_min_capability(info, 7, 5));
  KF_EXPECT_TRUE(!kernelforge::meets_min_capability(info, 9, 0));
  KF_EXPECT_TRUE(!kernelforge::meets_min_capability(info, 8, 10));
}

KF_TEST(DeviceInfo, RequireCapabilityThrowsOnUnmetRequirement) {
  DeviceInfo info;
  info.name = "Synthetic Test Device";
  info.compute_capability_major = 7;
  info.compute_capability_minor = 0;

  bool threw = false;
  try {
    kernelforge::require_capability_or_throw(info, 8, 9);
  } catch (const std::exception&) {
    threw = true;
  }
  KF_EXPECT_TRUE(threw);
}

KF_TEST(DeviceInfo, RequireCapabilityPassesOnRealDevice) {
  const DeviceInfo info = kernelforge::query_device_info(0);
  bool threw = false;
  try {
    kernelforge::require_capability_or_throw(info, 8, 9); // ADR 0002 minimum
  } catch (const std::exception&) {
    threw = true;
  }
  KF_EXPECT_TRUE(!threw);
}

KF_TEST(DeviceInfo, JsonAndHumanReadableAreNonEmptyAndParseableShape) {
  const DeviceInfo info = kernelforge::query_device_info(0);
  const std::string json = kernelforge::to_json(info);
  const std::string human = kernelforge::to_human_readable(info);
  KF_EXPECT_TRUE(json.front() == '{');
  KF_EXPECT_TRUE(json.back() == '}');
  KF_EXPECT_TRUE(json.find("\"compute_capability_major\":8") != std::string::npos);
  KF_EXPECT_TRUE(human.find("RTX 4090") != std::string::npos ||
                 human.find("Name") != std::string::npos);
}
