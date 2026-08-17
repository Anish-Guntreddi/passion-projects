# Generates src/common/build_info.generated.hpp so every benchmark result
# can self-report the exact toolchain/flags used to produce it (FR4:
# "compiler flags", "CUDA/toolkit/compiler versions"), without hand-copying
# that information into source and letting it drift.
set(KF_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${KF_GENERATED_DIR}")

string(TIMESTAMP KF_CONFIGURE_TIME_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/build_info.generated.hpp.in"
  "${KF_GENERATED_DIR}/build_info.generated.hpp"
  @ONLY
)

# Exposed as an interface library so any target can
# target_link_libraries(... PRIVATE kf_build_info) to get the include dir.
add_library(kf_build_info INTERFACE)
target_include_directories(kf_build_info INTERFACE "${KF_GENERATED_DIR}")
