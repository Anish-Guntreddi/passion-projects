# Opt-in sanitizer wiring, applied via an INTERFACE target.
#
# ASan+UBSan may be combined; TSan is mutually exclusive with ASan (both
# instrument memory access and cannot coexist in one binary). Sanitizer
# builds are selected at configure time (see scripts/build.sh) because
# instrumentation must be compiled into every translation unit, including
# GoogleTest itself.
function(arcserve_enable_sanitizers target_name)
  set(sanitizers "")

  if(ARCSERVE_ENABLE_ASAN)
    list(APPEND sanitizers "address")
  endif()

  if(ARCSERVE_ENABLE_UBSAN)
    list(APPEND sanitizers "undefined")
  endif()

  if(ARCSERVE_ENABLE_TSAN)
    if(ARCSERVE_ENABLE_ASAN)
      message(FATAL_ERROR "ARCSERVE_ENABLE_TSAN cannot be combined with ARCSERVE_ENABLE_ASAN")
    endif()
    list(APPEND sanitizers "thread")
  endif()

  if(sanitizers)
    list(JOIN sanitizers "," sanitizers_joined)
    target_compile_options(${target_name} INTERFACE
      -fsanitize=${sanitizers_joined}
      -fno-omit-frame-pointer
      -g
    )
    target_link_options(${target_name} INTERFACE -fsanitize=${sanitizers_joined})
  endif()
endfunction()
