add_executable(kf_tests
  test_main.cpp
  test_device_info.cpp
  test_vector_add.cpp
  test_saxpy.cpp
  test_transpose.cpp
  test_stride_copy.cpp
  test_bench_schema.cpp
  test_launch_validate.cpp
  test_bench_cli.cpp
  test_reduction.cpp
  test_scan.cpp
  test_histogram.cpp
)

target_include_directories(kf_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(kf_tests PRIVATE kf_common kf_kernels CUDA::cudart kf_warnings)

add_test(NAME kf_tests COMMAND kf_tests)
set_tests_properties(kf_tests PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED TESTS:")
