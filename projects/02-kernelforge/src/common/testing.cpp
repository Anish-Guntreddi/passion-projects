#include "common/testing.hpp"

#include <cstring>
#include <exception>

namespace kernelforge::testing {

int run_all(int argc, char** argv) {
  std::string filter;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const std::string prefix = "--filter=";
    if (arg.rfind(prefix, 0) == 0) {
      filter = arg.substr(prefix.size());
    }
  }

  int total = 0;
  int passed = 0;
  std::vector<std::string> failed_names;

  for (auto& tc : registry()) {
    const std::string full_name = tc.suite + "." + tc.name;
    if (!filter.empty() && full_name.find(filter) == std::string::npos) {
      continue;
    }
    ++total;
    std::printf("[ RUN      ] %s\n", full_name.c_str());
    current_failures().clear();

    bool fatal = false;
    try {
      tc.fn();
    } catch (const FatalTestFailure&) {
      fatal = true;
    } catch (const std::exception& e) {
      record_failure(std::string("uncaught exception: ") + e.what());
    } catch (...) {
      record_failure("uncaught non-std::exception");
    }

    if (current_failures().empty()) {
      std::printf("[       OK ] %s\n", full_name.c_str());
      ++passed;
    } else {
      std::printf("[  FAILED  ] %s (%zu failure(s)%s)\n", full_name.c_str(),
                  current_failures().size(), fatal ? ", fatal" : "");
      failed_names.push_back(full_name);
    }
  }

  std::printf("------------------------------------------------------------\n");
  std::printf("%d/%d tests passed\n", passed, total);
  if (!failed_names.empty()) {
    std::printf("FAILED TESTS:\n");
    for (const auto& n : failed_names) {
      std::printf("  - %s\n", n.c_str());
    }
    return 1;
  }
  return 0;
}

} // namespace kernelforge::testing
