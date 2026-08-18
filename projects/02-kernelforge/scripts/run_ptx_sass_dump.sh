#!/usr/bin/env bash
# Regenerates profiling/ptx-sass/{transpose,reduction,gemm}.{ptx,sass}.txt
# (Phase 6, ADR 0014/0006) from this repo's own compiled benchmark
# binaries via cuobjdump -- a static disassembler, needs no GPU access, so
# (unlike scripts/run_bench_*.sh or run_nsys_case_studies.sh) this needs
# neither the GPU-benchmark repo lock nor even a working GPU driver.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_DIR="${ROOT_DIR}/profiling/ptx-sass"

# cuobjdump ships under the CUDA toolkit's bin/ but is not always on
# $PATH (ADR 0014's finding for ncu applies here too -- verify with the
# same explicit-path pattern this repo's cmake/CudaOptions.cmake uses for
# nvcc).
CUOBJDUMP="/usr/local/cuda/bin/cuobjdump"
if ! [ -x "${CUOBJDUMP}" ]; then
  CUOBJDUMP="cuobjdump" # fall back to $PATH if the pinned path doesn't exist
fi

"${ROOT_DIR}/scripts/build.sh"
mkdir -p "${OUT_DIR}"

"${CUOBJDUMP}" --dump-ptx  "${BUILD_DIR}/apps/bench_transpose" > "${OUT_DIR}/transpose.ptx.txt"
"${CUOBJDUMP}" --dump-sass "${BUILD_DIR}/apps/bench_transpose" > "${OUT_DIR}/transpose.sass.txt"
"${CUOBJDUMP}" --dump-ptx  "${BUILD_DIR}/apps/bench_reduction" > "${OUT_DIR}/reduction.ptx.txt"
"${CUOBJDUMP}" --dump-sass "${BUILD_DIR}/apps/bench_reduction" > "${OUT_DIR}/reduction.sass.txt"
"${CUOBJDUMP}" --dump-ptx  "${BUILD_DIR}/apps/bench_gemm"      > "${OUT_DIR}/gemm.ptx.txt"
"${CUOBJDUMP}" --dump-sass "${BUILD_DIR}/apps/bench_gemm"      > "${OUT_DIR}/gemm.sass.txt"

echo "run_ptx_sass_dump: wrote $(ls "${OUT_DIR}"/*.txt | wc -l) files under ${OUT_DIR}"
