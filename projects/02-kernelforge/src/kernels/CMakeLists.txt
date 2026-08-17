add_library(kf_kernels STATIC
  vector/vector_add.cu
  vector/saxpy.cu
  transpose/transpose_naive.cu
  transpose/transpose_tiled.cu
  microbench/stride_copy.cu
  reduction/reduce_naive_global.cu
  reduction/reduce_naive_interleaved.cu
  reduction/reduce_sequential_addressing.cu
  reduction/reduce_warp_shuffle.cu
  reduction/reduce_vectorized_coarsened.cu
  scan/scan_common.cu
  scan/scan_hillis_steele.cu
  scan/scan_blelloch.cu
  histogram/histogram_global_atomic.cu
  histogram/histogram_privatized.cu
  histogram/histogram_privatized_coarsened.cu
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
