# Shared test-only support code (temp file / truncation / corruption
# helpers), reused by the recovery and corruption test binaries.
add_library(pebbledb_test_support STATIC
  support/temp_file.cpp
)
target_include_directories(pebbledb_test_support PUBLIC support)
target_link_libraries(pebbledb_test_support PUBLIC pebbledb_core)

add_executable(pebbledb_unit_tests
  unit/test_status.cpp
  unit/test_slice.cpp
  unit/test_coding.cpp
  unit/test_crc32c.cpp
  unit/test_db_map_reference.cpp
  unit/test_db_concurrency.cpp
  unit/test_wal_record_codec.cpp
)
target_link_libraries(pebbledb_unit_tests PRIVATE
  pebbledb_core
  GTest::gtest
  GTest::gtest_main
)

add_executable(pebbledb_recovery_tests
  recovery/test_wal_truncation_recovery.cpp
)
target_link_libraries(pebbledb_recovery_tests PRIVATE
  pebbledb_core
  pebbledb_test_support
  GTest::gtest
  GTest::gtest_main
)

add_executable(pebbledb_corruption_tests
  corruption/test_wal_corruption.cpp
)
target_link_libraries(pebbledb_corruption_tests PRIVATE
  pebbledb_core
  pebbledb_test_support
  GTest::gtest
  GTest::gtest_main
)

# WSL2's kernel maps some region ThreadSanitizer's runtime doesn't expect
# ("FATAL: ThreadSanitizer: unexpected memory mapping ..."), avoided by
# disabling ASLR for the process (a documented WSL2/TSan interaction, not
# a PebbleDB bug — same workaround as the portfolio's ArcServe project).
# setarch -R does this without needing root. CROSSCOMPILING_EMULATOR makes
# both gtest_discover_tests' own discovery run (a build-time step) and
# every ctest invocation go through it. Harmless no-op on a real Linux
# host where TSan doesn't need it, but scoped to TSan builds only.
if(PEBBLEDB_ENABLE_TSAN)
  set_target_properties(pebbledb_unit_tests pebbledb_recovery_tests pebbledb_corruption_tests
    PROPERTIES
      CROSSCOMPILING_EMULATOR "setarch;${CMAKE_HOST_SYSTEM_PROCESSOR};-R"
  )
endif()

include(GoogleTest)
gtest_discover_tests(pebbledb_unit_tests)
# The exhaustive per-byte truncation test and the 5000-iteration
# randomized DB test are the slowest cases here; 60s is generous headroom
# for a debug or sanitizer-instrumented build.
gtest_discover_tests(pebbledb_recovery_tests PROPERTIES TIMEOUT 60)
gtest_discover_tests(pebbledb_corruption_tests PROPERTIES TIMEOUT 60)
