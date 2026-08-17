# Project-wide warning flags, applied via an INTERFACE target so every
# consumer (library, binary, tests) gets identical settings.
#
# Deliberately excluded: -Wconversion / -Wsign-conversion / -Wold-style-cast.
# Raw POSIX socket code (sockaddr casts, size_t<->ssize_t/int traffic from
# recv()/send()/read()) trips these constantly without a corresponding
# correctness benefit at this project's scope; see
# docs/decisions/0001-build-tooling-and-test-framework.md for the tradeoff.
function(arcserve_set_project_warnings target_name)
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

  if(ARCSERVE_WARNINGS_AS_ERRORS)
    list(APPEND clang_gcc_warnings -Werror)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target_name} INTERFACE ${clang_gcc_warnings})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(msvc_warnings /W4)
    if(ARCSERVE_WARNINGS_AS_ERRORS)
      list(APPEND msvc_warnings /WX)
    endif()
    target_compile_options(${target_name} INTERFACE ${msvc_warnings})
  else()
    message(WARNING "Unrecognized compiler '${CMAKE_CXX_COMPILER_ID}' — no warning flags applied")
  endif()
endfunction()
