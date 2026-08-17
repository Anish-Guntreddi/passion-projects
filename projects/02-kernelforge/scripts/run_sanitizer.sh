#!/usr/bin/env bash
# Compute Sanitizer workflow (FR6). Runs the memcheck tool (out-of-bounds /
# misaligned device memory access detection) over the correctness test
# binary, which exercises every kernel in this repo across many sizes
# including edge cases. Explicit binary path (not relying on $PATH) per
# this repo's WSL2 quoting/PATH conventions.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
SANITIZER="/usr/local/cuda/bin/compute-sanitizer"

if [ ! -x "${SANITIZER}" ]; then
  echo "compute-sanitizer not found at ${SANITIZER}" >&2
  exit 1
fi

TOOL="${1:-memcheck}" # memcheck | racecheck | initcheck | synccheck
shift || true

"${SANITIZER}" --tool "${TOOL}" "${BUILD_DIR}/tests/kf_tests" "$@"
