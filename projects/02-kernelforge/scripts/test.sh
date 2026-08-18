#!/usr/bin/env bash
# Builds (if needed), runs the correctness test suite, and validates every
# committed benchmark result against the JSON Schema.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

"${ROOT_DIR}/scripts/build.sh"

echo "=== kf_tests ==="
"${BUILD_DIR}/tests/kf_tests" "$@"

echo ""
echo "=== validating benchmarks/raw/*.jsonl against schema ==="
python3 "${ROOT_DIR}/scripts/validate_results.py" \
  "${ROOT_DIR}/benchmarks/raw" \
  "${ROOT_DIR}/benchmarks/schema/bench_result.schema.json"

echo ""
echo "=== validating profiling/occupancy/*.jsonl against schema (Phase 6, ADR 0014) ==="
python3 "${ROOT_DIR}/scripts/validate_occupancy.py" \
  "${ROOT_DIR}/profiling/occupancy" \
  "${ROOT_DIR}/profiling/schema/occupancy_report.schema.json"
