add_library(kf_kernels STATIC
  vector/vector_add.cu
  vector/saxpy.cu
  transpose/transpose_naive.cu
  transpose/transpose_tiled.cu
  microbench/stride_copy.cu
)

target_include_directories(kf_kernels PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(kf_kernels
  PUBLIC
    CUDA::cudart
    kf_common
  PRIVATE
    kf_cuda_options
)

set_target_properties(kf_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)
