#!/usr/bin/env bash
# Create the WSL venv for MiniPaged and install dependencies REPRODUCIBLY.
#
# Usage (from WSL, project root or anywhere):
#   bash scripts/setup_env.sh                 # default: install the exact
#                                               # pinned versions recorded in
#                                               # requirements-lock.txt
#   bash scripts/setup_env.sh --update-lock    # resolve fresh versions from
#                                               # pyproject.toml's ranges and
#                                               # regenerate the lock file
#
# Why two modes: a plain `pip install -e ".[dev]"` against pyproject.toml's
# `>=` version ranges lets a fresh clone resolve different package versions
# than the machine the project was developed/tested on, silently breaking
# reproducibility. The default mode below installs `requirements-lock.txt`
# verbatim with `pip install -r`, so a fresh clone gets *exactly* the
# versions this project was verified against. Re-resolving newer versions
# is still possible, but only via the explicit --update-lock action.
#
# Phases 0-1 are CPU-only, pure-Python domain simulation (no torch/CUDA
# dependency yet -- see docs/decisions/0001-simulation-first-architecture.md).
set -euo pipefail

cd "$(dirname "$0")/.."

PIP_VERSION="24.0"
LOCK_FILE="requirements-lock.txt"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade "pip==${PIP_VERSION}"

if [[ "${1:-}" == "--update-lock" ]]; then
  echo "== Resolving fresh dependency versions (not reproducible by itself) =="
  pip install -e ".[dev]"

  # Exclude minipaged's own editable-install line: `pip freeze` records it
  # as a VCS/path reference tied to one checkout, meaningless for a fresh
  # clone (it's installed via `pip install -e . --no-deps` below instead).
  pip freeze | grep -v 'egg=minipaged' > "$LOCK_FILE"

  echo
  echo "Regenerated $(pwd)/${LOCK_FILE}."
  echo "Review the diff and commit it deliberately -- this is the only step"
  echo "that is allowed to change pinned dependency versions."
else
  if [[ ! -f "$LOCK_FILE" ]]; then
    echo "error: ${LOCK_FILE} not found." >&2
    echo "Run 'bash scripts/setup_env.sh --update-lock' once to create it." >&2
    exit 1
  fi

  echo "== Installing pinned dependencies from ${LOCK_FILE} (reproducible) =="
  pip install -r "$LOCK_FILE"
  # --no-deps: dependencies are already fully pinned above; this only adds
  # minipaged's own package (editable) without letting pip re-resolve anything.
  pip install -e . --no-deps
fi

echo
echo "Environment ready at $(pwd)/.venv"
echo "Activate with: source .venv/bin/activate"
