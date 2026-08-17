# Host-compiler warning flags. CUDA device code gets a lighter touch since
# nvcc/host-compiler warning interaction is noisier and not the focus here.
add_library(kf_warnings INTERFACE)

target_compile_options(kf_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

if(KF_WARNINGS_AS_ERRORS)
  target_compile_options(kf_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
  )
endif()
