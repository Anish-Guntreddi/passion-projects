// `device-info` command (FR3). Prints the queried device's properties
// (human-readable by default, --json for machine-readable), and performs
// the capability check every other executable in this repo performs
// before doing any GPU work (hard constraint 8: fail loudly).
#include <cstdio>
#include <cstring>
#include <exception>

#include "common/device_query.hpp"

namespace {
constexpr int kRequiredMajor = 8; // ADR 0002
constexpr int kRequiredMinor = 9;
} // namespace

int main(int argc, char** argv) {
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) json = true;
  }

  try {
    const kernelforge::DeviceInfo info = kernelforge::query_device_info(0);

    if (json) {
      std::printf("%s\n", kernelforge::to_json(info).c_str());
    } else {
      std::printf("%s", kernelforge::to_human_readable(info).c_str());
    }

    kernelforge::require_capability_or_throw(info, kRequiredMajor, kRequiredMinor);
    if (!json) {
      std::printf("\nCapability check: OK (>= %d.%d required, see ADR 0002)\n",
                   kRequiredMajor, kRequiredMinor);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "device-info: FATAL: %s\n", e.what());
    return 1;
  }
}
