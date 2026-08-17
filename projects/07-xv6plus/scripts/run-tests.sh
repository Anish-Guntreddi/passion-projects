#!/usr/bin/env bash
# xv6-plus: build + run the full test suite in QEMU.
# See scripts/run_tests.py for stage-by-stage documentation and flags.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 scripts/run_tests.py "$@"
