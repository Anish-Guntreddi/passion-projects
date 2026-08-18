#!/usr/bin/env bash
# Regenerates profiling/occupancy/{transpose,reduction,gemm}.jsonl exactly
# (Phase 6, ADR 0014) -- the occupancy_report CLI's output is deterministic
# (computed from the compiled kernel's cubin + this device's static limits,
# not affected by GPU load/clocks the way a timing benchmark is), so
# re-running this script should reproduce byte-identical numeric fields to
# what is committed, on the same GPU/toolchain/build.
#
# Unlike scripts/run_bench_*.sh, this does NOT need the GPU-benchmark repo
# lock (no wall-clock timing is measured) and builds if needed via
# scripts/build.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/apps/occupancy_report"
OUT_DIR="${ROOT_DIR}/profiling/occupancy"

"${ROOT_DIR}/scripts/build.sh"
mkdir -p "${OUT_DIR}"

# transpose: both variants at n=8192 (Phase 1's largest, fully-HBM-bound
# benchmarked size -- see benchmarks/methodology.md §8 / profiling/
# case-studies/01-transpose-bank-conflict.md).
: > "${OUT_DIR}/transpose.jsonl"
"${BIN}" --family transpose --variant v1_naive --n 8192 >> "${OUT_DIR}/transpose.jsonl"
"${BIN}" --family transpose --variant v2_tiled  --n 8192 >> "${OUT_DIR}/transpose.jsonl"

# reduction: V2/V3/V4 at block-size=256 (this ladder's benchmarked block
# size -- see benchmarks/methodology.md §9.1 / profiling/case-studies/
# 02-reduction-warp-shuffle-barriers.md).
: > "${OUT_DIR}/reduction.jsonl"
"${BIN}" --family reduction --variant v2_sequential_addressing --block-size 256 >> "${OUT_DIR}/reduction.jsonl"
"${BIN}" --family reduction --variant v3_warp_shuffle           --block-size 256 >> "${OUT_DIR}/reduction.jsonl"
"${BIN}" --family reduction --variant v4_vectorized_coarsened    --block-size 256 >> "${OUT_DIR}/reduction.jsonl"

# gemm: V3/V4 across the same 5-size sweep as benchmarks/configs/gemm.json
# (see benchmarks/methodology.md §11 / profiling/case-studies/
# 03-gemm-register-tiling-occupancy.md for the grid-utilization crossover
# this sweep exists to demonstrate).
: > "${OUT_DIR}/gemm.jsonl"
for sz in 128 256 512 1024 2048; do
  "${BIN}" --family gemm --variant v3_tiled --m "${sz}" --n "${sz}" >> "${OUT_DIR}/gemm.jsonl"
done
for sz in 128 256 512 1024 2048; do
  "${BIN}" --family gemm --variant v4_register_tiled --m "${sz}" --n "${sz}" >> "${OUT_DIR}/gemm.jsonl"
done

echo ""
echo "=== validating profiling/occupancy/*.jsonl against schema ==="
python3 "${ROOT_DIR}/scripts/validate_occupancy.py" \
  "${OUT_DIR}" \
  "${ROOT_DIR}/profiling/schema/occupancy_report.schema.json"
