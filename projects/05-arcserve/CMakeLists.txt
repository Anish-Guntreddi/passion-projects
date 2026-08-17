cmake_minimum_required(VERSION 3.20)
project(arcserve LANGUAGES CXX)

# --- Language standard -------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# clangd / clang-tidy need this to see include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(WARNING
    "ArcServe targets Linux (epoll, POSIX sockets). "
    "Configuring on ${CMAKE_SYSTEM_NAME} is unsupported for anything past "
    "the header-only pieces.")
endif()

# --- Project-wide options ----------------------------------------------
option(ARCSERVE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(ARCSERVE_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(ARCSERVE_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(ARCSERVE_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(ARCSERVE_BUILD_TESTS "Build unit/integration tests" ON)

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

add_library(arcserve_warnings INTERFACE)
arcserve_set_project_warnings(arcserve_warnings)

add_library(arcserve_sanitizers INTERFACE)
arcserve_enable_sanitizers(arcserve_sanitizers)

# --- Core library --------------------------------------------------------
# Everything reusable (net primitives, protocol parser, server loop) lives
# in arcserve_core so both the production binary and the test binaries link
# the exact same translation units.
add_library(arcserve_core
  src/app/cli.cpp
  src/buffers/byte_buffer.cpp
  src/concurrency/bounded_work_queue.cpp
  src/concurrency/worker_pool.cpp
  src/net/socket.cpp
  src/protocol/http_message.cpp
  src/protocol/http_parser.cpp
  src/reactor/connection.cpp
  src/reactor/epoll_reactor.cpp
  src/reactor/output_queue.cpp
  src/server/blocking_server.cpp
  src/server/default_handlers.cpp
  src/server/echo_reactor_server.cpp
  src/server/nonblocking_http_server.cpp
)
target_include_directories(arcserve_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(arcserve_core
  PUBLIC
    arcserve_warnings
    arcserve_sanitizers
)
target_compile_features(arcserve_core PUBLIC cxx_std_20)

if(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(arcserve_core PUBLIC Threads::Threads)
endif()

# --- Executable ----------------------------------------------------------
add_executable(arcserve_server src/app/main.cpp)
target_link_libraries(arcserve_server PRIVATE arcserve_core)

# --- Tests -----------------------------------------------------------------
if(ARCSERVE_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  # Match the CRT/exception settings of the rest of the build; see
  # docs/decisions/0001-build-tooling-and-test-framework.md.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()
