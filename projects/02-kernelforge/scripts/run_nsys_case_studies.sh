#!/usr/bin/env bash
# Regenerates the 6 profiling/nsight-systems/*.nsys-rep captures (+ a
# `nsys stats` text summary alongside each) used by the 3 Phase 6 case
# studies (ADR 0014). These DO run real kernels under a timed capture, so
# -- like scripts/run_bench_*.sh -- this needs the repo-wide GPU-benchmark
# lock held for the duration if its output will be cited in docs/README
# (per this repo's orchestration convention; see the lock acquire/release
# commands documented alongside that convention). It is not wrapped here
# automatically since the lock is a cross-project convention this script
# does not own.
#
# Every capture's GPU-side kernel/memory trace section will be empty
# (`nsys stats` reports `SKIPPED: ... does not contain CUDA kernel data`)
# -- this is expected, not a bug in this script; see ADR 0014. Only the
# host-side CUDA API summary (cuda_api_sum) is populated.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_DIR="${ROOT_DIR}/profiling/nsight-systems"

"${ROOT_DIR}/scripts/build.sh"
mkdir -p "${OUT_DIR}"

capture() {
  local out_name="$1"; shift
  nsys profile -o "${OUT_DIR}/${out_name}" --force-overwrite=true "$@"
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    "${OUT_DIR}/${out_name}.nsys-rep" > "${OUT_DIR}/${out_name}.stats.txt" 2>&1
  rm -f "${OUT_DIR}/${out_name}.sqlite" # regenerable intermediate, not committed (.gitignore)
}

capture 01-transpose-v1-naive \
  "${BUILD_DIR}/apps/bench_transpose" --variant v1_naive --n 8192 --warmup 10 --reps 30
capture 01-transpose-v2-tiled \
  "${BUILD_DIR}/apps/bench_transpose" --variant v2_tiled --n 8192 --warmup 10 --reps 30

capture 02-reduction-v2-sequential \
  "${BUILD_DIR}/apps/bench_reduction" --variant v2_sequential_addressing --n 67108864 --warmup 10 --reps 30
capture 02-reduction-v3-warpshuffle \
  "${BUILD_DIR}/apps/bench_reduction" --variant v3_warp_shuffle --n 67108864 --warmup 10 --reps 30

capture 03-gemm-v3-tiled \
  "${BUILD_DIR}/apps/bench_gemm" --variant v3_tiled --n 2048 --warmup 10 --reps 30
capture 03-gemm-v4-register-tiled \
  "${BUILD_DIR}/apps/bench_gemm" --variant v4_register_tiled --n 2048 --warmup 10 --reps 30

echo "run_nsys_case_studies: wrote $(ls "${OUT_DIR}"/*.nsys-rep | wc -l) .nsys-rep captures under ${OUT_DIR}"
