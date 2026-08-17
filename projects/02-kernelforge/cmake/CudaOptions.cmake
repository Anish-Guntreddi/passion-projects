# CUDA compile options shared by every target that compiles .cu sources.
#
# -lineinfo: embeds source line info in the generated binary so Nsight
#   Compute/Systems (Phase 6) can correlate profiler samples back to CUDA
#   source lines, at negligible runtime cost. Kept on for every build type
#   since profiling is a first-class use case of this repo.
# No --use_fast_math: correctness (FR6 tolerances) matters more than a few
#   percent of speed for FP32 add/multiply-heavy kernels in Phases 0-1;
#   revisit per-kernel if a later phase's hypothesis is specifically about
#   fast-math tradeoffs.
add_library(kf_cuda_options INTERFACE)

target_compile_options(kf_cuda_options INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
