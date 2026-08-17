# Common infrastructure library: error macros, timers, device info, RNG,
# reference implementations, comparison, stats, benchmark schema, and the
# lightweight test framework (spec repo layout: src/common/).
add_library(kf_common STATIC
  error_check.cpp
  device_query.cu
  reference_ops.cpp
  compare.cpp
  stats.cpp
  bench_result.cpp
  testing.cpp
)

target_include_directories(kf_common PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(kf_common
  PUBLIC
    CUDA::cudart
  PRIVATE
    kf_warnings
    kf_cuda_options
    kf_build_info
)

set_target_properties(kf_common PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)
