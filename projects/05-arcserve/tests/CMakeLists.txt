# Test-only support code (raw socket test client), reused by every
# integration test binary.
add_library(arcserve_test_support STATIC
  support/test_socket_client.cpp
)
target_include_directories(arcserve_test_support PUBLIC support)
target_link_libraries(arcserve_test_support PUBLIC arcserve_core)

add_executable(arcserve_unit_tests
  unit/test_cli.cpp
  unit/test_connection.cpp
  unit/test_epoll_reactor.cpp
  unit/test_file_descriptor.cpp
  unit/test_http_message.cpp
  unit/test_http_parser.cpp
  unit/test_socket.cpp
)
target_link_libraries(arcserve_unit_tests PRIVATE
  arcserve_core
  GTest::gtest
  GTest::gtest_main
)

add_executable(arcserve_integration_tests
  integration/test_blocking_server_integration.cpp
  integration/test_echo_reactor_server_integration.cpp
  integration/test_nonblocking_http_server_integration.cpp
)
target_link_libraries(arcserve_integration_tests PRIVATE
  arcserve_core
  arcserve_test_support
  GTest::gtest
  GTest::gtest_main
)

# WSL2's kernel maps some region ThreadSanitizer's runtime doesn't expect
# ("FATAL: ThreadSanitizer: unexpected memory mapping ..."), which happens
# to be avoided by disabling ASLR for the process (a documented WSL2/TSan
# interaction, not an ArcServe bug). setarch -R does this without needing
# root. CROSSCOMPILING_EMULATOR makes both gtest_discover_tests' own
# discovery run (a build-time step) and every ctest invocation go through
# it. Harmless no-op on a real Linux host where TSan doesn't need it, but
# scoped to TSan builds only to avoid adding an unnecessary indirection
# everywhere else.
if(ARCSERVE_ENABLE_TSAN)
  set_target_properties(arcserve_unit_tests arcserve_integration_tests PROPERTIES
    CROSSCOMPILING_EMULATOR "setarch;${CMAKE_HOST_SYSTEM_PROCESSOR};-R"
  )
endif()

include(GoogleTest)
# TIMEOUT here matches the integration-test precedent below: it exists so a
# hang-reproducing regression test (e.g.
# EpollReactorTest.StopRequestedBeforeRunStillReturnsPromptly) fails loudly
# within a bounded time instead of hanging ctest/CI forever if the bug it
# guards against is ever reintroduced.
gtest_discover_tests(arcserve_unit_tests PROPERTIES TIMEOUT 30)
gtest_discover_tests(arcserve_integration_tests PROPERTIES TIMEOUT 30)
