# Project-wide warning flags, applied via an INTERFACE target so every
# consumer (library, tools, tests) gets identical settings.
#
# Deliberately excluded: -Wconversion / -Wsign-conversion / -Wold-style-cast.
# The WAL/SSTable byte-level I/O code (checksum computation, fixed-width
# length/sequence encoding, POSIX read()/write() size_t<->ssize_t traffic)
# trips these constantly without a corresponding correctness benefit at
# this project's scope; see
# docs/decisions/0001-build-tooling-and-warnings.md for the tradeoff. This
# mirrors the equivalent decision in the portfolio's ArcServe project for
# the same reason (raw byte/socket code, not application logic).
function(pebbledb_set_project_warnings target_name)
  set(clang_gcc_warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wformat=2
    -Wimplicit-fallthrough
    -Wunused
  )

  if(PEBBLEDB_WARNINGS_AS_ERRORS)
    list(APPEND clang_gcc_warnings -Werror)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target_name} INTERFACE ${clang_gcc_warnings})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(msvc_warnings /W4)
    if(PEBBLEDB_WARNINGS_AS_ERRORS)
      list(APPEND msvc_warnings /WX)
    endif()
    target_compile_options(${target_name} INTERFACE ${msvc_warnings})
  else()
    message(WARNING "Unrecognized compiler '${CMAKE_CXX_COMPILER_ID}' — no warning flags applied")
  endif()
endfunction()
