#!/usr/bin/env bash
# Runs BOTH contention profiles (Phase 3 exit criterion: low- vs high-
# contention workloads documented). Two separate config files/output
# files rather than a single sweep, since scripts/bench_driver.py sweeps
# one dimension (variant x sweep_param) at a time and contention is a
# second independent dimension -- see benchmarks/configs/histogram_*.json.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${ROOT_DIR}/scripts/bench_driver.py" \
  --config "${ROOT_DIR}/benchmarks/configs/histogram_uniform.json" \
  --bin-dir "${ROOT_DIR}/build/apps" \
  --out "${ROOT_DIR}/benchmarks/raw/histogram_uniform.jsonl"

python3 "${ROOT_DIR}/scripts/bench_driver.py" \
  --config "${ROOT_DIR}/benchmarks/configs/histogram_skewed.json" \
  --bin-dir "${ROOT_DIR}/build/apps" \
  --out "${ROOT_DIR}/benchmarks/raw/histogram_skewed.jsonl"
