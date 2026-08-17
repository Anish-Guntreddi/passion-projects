#!/usr/bin/env bash
# Formats all project sources in place with clang-format (config: .clang-format).
# Not currently runnable in this project's WSL2 dev environment (clang-format
# is not installed there and passwordless sudo is unavailable) — see
# docs/decisions/0001-build-tooling-and-test-framework.md. CI installs it
# fresh, so this script is what CI's format-check job runs (with --dry-run).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${1:-}"

mapfile -t FILES < <(find "${ROOT_DIR}/include" "${ROOT_DIR}/src" "${ROOT_DIR}/tests" \
  -type f \( -name '*.hpp' -o -name '*.cpp' \))

if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  clang-format --dry-run --Werror "${FILES[@]}"
else
  clang-format -i "${FILES[@]}"
fi
