#!/usr/bin/env bash
# Builds (if needed) and runs the full test suite. Usage:
#   scripts/test.sh [BuildType] [asan|ubsan|tsan|none]
set -euo pipefail

BUILD_TYPE="${1:-Debug}"
SANITIZER="${2:-none}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/${BUILD_TYPE}-${SANITIZER}"

"${ROOT_DIR}/scripts/build.sh" "${BUILD_TYPE}" "${SANITIZER}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
