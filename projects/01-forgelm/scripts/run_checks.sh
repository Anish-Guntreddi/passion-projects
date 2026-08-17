#!/usr/bin/env bash
# Run the exact lint + type + test sequence CI runs. Use this locally before
# pushing, and as the source of truth if the GitHub Actions workflow and
# this script ever drift.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== ruff (lint) =="
ruff check src tests

echo "== ruff (format check) =="
ruff format --check src tests

echo "== pyright (types) =="
pyright

echo "== pytest (unit + integration) =="
pytest -v
