#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT_DIR}/scripts/bench_driver.py" \
  --config "${ROOT_DIR}/benchmarks/configs/gemm.json" \
  --bin-dir "${ROOT_DIR}/build/apps" \
  --out "${ROOT_DIR}/benchmarks/raw/gemm.jsonl"
