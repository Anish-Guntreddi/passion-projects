#!/usr/bin/env bash
# Runs the full FlashLite test suite (Phase 0-5) + benchmark-schema
# validation. Must run inside WSL2 Ubuntu, inside a venv with the package
# installed (scripts/build_ext.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pytest -v
python scripts/validate_results.py --allow-empty
