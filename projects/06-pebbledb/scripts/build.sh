#!/usr/bin/env bash
# Configures and builds PebbleDB. Usage:
#   scripts/build.sh [BuildType] [asan|ubsan|tsan|none]
#
# Examples:
#   scripts/build.sh                 # Debug, no sanitizer
#   scripts/build.sh Debug asan      # Debug + ASan/UBSan
#   scripts/build.sh Release none    # Release, no sanitizer
set -euo pipefail

BUILD_TYPE="${1:-Debug}"
SANITIZER="${2:-none}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/${BUILD_TYPE}-${SANITIZER}"

CMAKE_ARGS=(
  -G Ninja
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

case "${SANITIZER}" in
  asan)
    CMAKE_ARGS+=(-DPEBBLEDB_ENABLE_ASAN=ON -DPEBBLEDB_ENABLE_UBSAN=ON)
    ;;
  ubsan)
    CMAKE_ARGS+=(-DPEBBLEDB_ENABLE_UBSAN=ON)
    ;;
  tsan)
    CMAKE_ARGS+=(-DPEBBLEDB_ENABLE_TSAN=ON)
    ;;
  none)
    ;;
  *)
    echo "Unknown sanitizer option: '${SANITIZER}' (expected asan|ubsan|tsan|none)" >&2
    exit 1
    ;;
esac

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel
