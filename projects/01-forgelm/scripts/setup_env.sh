#!/usr/bin/env bash
# Create the WSL venv for ForgeLM and install dependencies REPRODUCIBLY.
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
# than the machine the project was developed/benchmarked on, silently
# breaking reproducibility (spec risk register, section 1.10). The default
# mode below installs `requirements-lock.txt` verbatim with `pip install
# -r`, so a fresh clone gets *exactly* the versions this project was
# verified against. Re-resolving newer versions is still possible, but only
# via the explicit --update-lock action, so "reproducible install" and
# "refresh pinned versions" can never be silently conflated.
#
# Per the project's D3 default, the CPU path always works: torch's cu124
# wheel bundles CPU support, so this script works whether or not a CUDA
# device is present. GPU acceleration is used automatically when available.
set -euo pipefail

cd "$(dirname "$0")/.."

# Pinned pip version, so `pip` itself doesn't silently drift across fresh
# clones either. Bump deliberately (alongside --update-lock) when needed.
PIP_VERSION="26.2.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
LOCK_FILE="requirements-lock.txt"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade "pip==${PIP_VERSION}"

if [[ "${1:-}" == "--update-lock" ]]; then
  echo "== Resolving fresh dependency versions (not reproducible by itself) =="
  pip install torch --index-url "$TORCH_INDEX_URL"
  pip install -e ".[dev]"

  # Exclude forgelm's own editable-install line: `pip freeze` records it as
  # a VCS reference tied to one historical commit, which is meaningless for
  # a fresh clone (it's installed via `pip install -e . --no-deps` below,
  # not from this lock file).
  pip freeze | grep -v 'egg=forgelm' > "$LOCK_FILE"

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
  # --extra-index-url (not --index-url): most pinned packages live on PyPI;
  # only the "+cu124" torch/nvidia-* build tags require the torch index, so
  # both must be searched for the pinned versions to resolve.
  pip install -r "$LOCK_FILE" --extra-index-url "$TORCH_INDEX_URL"
  # --no-deps: dependencies are already fully pinned above; this only adds
  # forgelm's own package (editable) without letting pip re-resolve anything.
  pip install -e . --no-deps
fi

echo
echo "Environment ready at $(pwd)/.venv"
echo "Activate with: source .venv/bin/activate"
