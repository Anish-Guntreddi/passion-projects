#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
  "${ROOT_DIR}/scripts/configure.sh"
fi

cmake --build "${BUILD_DIR}" -- -j"$(nproc)"
