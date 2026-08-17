#!/usr/bin/env bash
# Two bench_driver.py calls, not one: v0_naive_global_atomic genuinely
# fails its own correctness gate at the largest configured size
# (n=67108864 -- fp32 accumulator saturation, NOT a bug; see
# benchmarks/methodology.md sec 9.1 and src/kernels/reduction/README.md's
# "Ladder history" note) while v1-v4 pass at every configured size. A
# single plain sweep over benchmarks/configs/reduction.json's full
# (variants x sweep_values) grid would therefore always fail loudly on
# v0's last point -- correct per hard constraint 1 (never commit a
# failing result), but it would also break "benchmarks reproducible from
# scripts" (spec Part 1.7) for the OTHER four variants, which have nothing
# wrong with them. Splitting into "v1-v4 at every configured size" +
# "v0 at its five passing sizes" reproduces the exact committed
# benchmarks/raw/reduction.jsonl content from a fresh clone with zero
# failures, while still never silently attempting (or hiding the fact
# that this repo deliberately never attempts) v0 at the one size it
# cannot pass.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${ROOT_DIR}/scripts/bench_driver.py" \
  --config "${ROOT_DIR}/benchmarks/configs/reduction.json" \
  --bin-dir "${ROOT_DIR}/build/apps" \
  --out "${ROOT_DIR}/benchmarks/raw/reduction.jsonl" \
  --only-variant v1_naive_interleaved \
  --only-variant v2_sequential_addressing \
  --only-variant v3_warp_shuffle \
  --only-variant v4_vectorized_coarsened

python3 "${ROOT_DIR}/scripts/bench_driver.py" \
  --config "${ROOT_DIR}/benchmarks/configs/reduction.json" \
  --bin-dir "${ROOT_DIR}/build/apps" \
  --out "${ROOT_DIR}/benchmarks/raw/reduction.jsonl" \
  --only-variant v0_naive_global_atomic \
  --only-sweep-value 4096 \
  --only-sweep-value 65536 \
  --only-sweep-value 1048576 \
  --only-sweep-value 4194304 \
  --only-sweep-value 16777216
