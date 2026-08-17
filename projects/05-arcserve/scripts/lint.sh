#!/usr/bin/env bash
# Runs clang-tidy (config: .clang-tidy) against every project translation
# unit, using the compile_commands.json from a configured build directory.
# Not currently runnable in this project's WSL2 dev environment
# (clang-tidy is not installed there and passwordless sudo is unavailable)
# — see docs/decisions/0001-build-tooling-and-test-framework.md. CI
# installs it fresh and runs exactly this script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/lint"

cmake -G Ninja -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Debug >/dev/null

mapfile -t FILES < <(find "${ROOT_DIR}/src" -type f -name '*.cpp')

clang-tidy -p "${BUILD_DIR}" "${FILES[@]}"
