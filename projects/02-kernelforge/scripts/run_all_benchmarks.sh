#!/usr/bin/env bash
# Runs every Phase 0/1 benchmark sweep and appends results to
# benchmarks/raw/. Separate from scripts/build.sh/test.sh on purpose
# (hard constraint 9: "separate implementation commits from benchmark-
# result commits" -- this script is the one that produces the artifacts
# that go in a benchmark-result commit).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/build.sh"

"${ROOT_DIR}/scripts/run_bench_vector_add.sh"
"${ROOT_DIR}/scripts/run_bench_saxpy.sh"
"${ROOT_DIR}/scripts/run_bench_transpose.sh"
"${ROOT_DIR}/scripts/run_bench_stride.sh"

python3 "${ROOT_DIR}/scripts/validate_results.py" \
  "${ROOT_DIR}/benchmarks/raw" \
  "${ROOT_DIR}/benchmarks/schema/bench_result.schema.json"

echo ""
echo "All Phase 0/1 benchmarks complete. Raw results under benchmarks/raw/."
