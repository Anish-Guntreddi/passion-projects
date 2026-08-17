#!/usr/bin/env bash
# Configures the CMake build. nvcc is referenced by an explicit path (never
# via $PATH lookup) since PATH resolution for CUDA differs between
# interactive and non-interactive WSL shells on this machine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="${KF_BUILD_TYPE:-RelWithDebInfo}"
CUDA_ARCH="${KF_CUDA_ARCH:-89}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DKF_CUDA_ARCH="${CUDA_ARCH}"

echo "Configured build in ${BUILD_DIR} (build type: ${BUILD_TYPE}, arch: sm_${CUDA_ARCH})"
