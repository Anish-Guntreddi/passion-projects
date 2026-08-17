cmake_minimum_required(VERSION 3.20)
project(pebbledb LANGUAGES CXX)

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
    "PebbleDB targets Linux (POSIX open/read/write/fsync for the WAL). "
    "Configuring on ${CMAKE_SYSTEM_NAME} is unsupported for anything past "
    "the header-only pieces.")
endif()

# --- Project-wide options ----------------------------------------------
option(PEBBLEDB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(PEBBLEDB_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(PEBBLEDB_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(PEBBLEDB_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(PEBBLEDB_BUILD_TESTS "Build unit/recovery/corruption tests" ON)

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

add_library(pebbledb_warnings INTERFACE)
pebbledb_set_project_warnings(pebbledb_warnings)

add_library(pebbledb_sanitizers INTERFACE)
pebbledb_enable_sanitizers(pebbledb_sanitizers)

# --- Core library --------------------------------------------------------
# Everything (status/error model, byte encoding, checksums, WAL, the
# Phase-0 in-memory DB) lives in pebbledb_core so tools and every test
# binary link the exact same translation units.
add_library(pebbledb_core
  src/status.cpp
  src/db.cpp
  src/util/coding.cpp
  src/util/crc32c.cpp
  src/util/posix_file.cpp
  src/wal/record.cpp
  src/wal/writer.cpp
  src/wal/reader.cpp
)
target_include_directories(pebbledb_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(pebbledb_core
  PUBLIC
    pebbledb_warnings
    pebbledb_sanitizers
)
target_compile_features(pebbledb_core PUBLIC cxx_std_20)

if(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(pebbledb_core PUBLIC Threads::Threads)
endif()

# --- Tests -----------------------------------------------------------------
# No production executable yet: Phase 0/1 deliver the library (in-memory DB
# + standalone WAL module) per the roadmap; tools/pebble_cli arrives once
# persistence is wired end-to-end (see README "What's not here yet").
if(PEBBLEDB_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  # Match the CRT/exception settings of the rest of the build; see
  # docs/decisions/0001-build-tooling-and-warnings.md.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()
