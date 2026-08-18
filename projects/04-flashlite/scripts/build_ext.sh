#!/usr/bin/env bash
# Builds/installs the flashlite package, including the V1 naive CUDA
# extension (ADR 0005: pybind11 + torch.utils.cpp_extension route).
# Must run inside WSL2 Ubuntu, inside a venv that already has torch>=2.4
# installed with CUDA support (see README.md "Setup").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to this Python/torch'"
# --no-build-isolation: without it, pip's PEP 517 isolated build environment
# re-resolves `torch` from [build-system] requires against the default
# index (picking up whatever the latest torch release is, which may bundle
# a different CUDA runtime than the one actually installed here) instead of
# reusing the already-installed, version-matched torch. CUDAExtension's own
# build-time CUDA-version check then fails against that mismatched
# isolated-env torch even though the real environment is consistent.
pip install --no-build-isolation -e ".[dev]"
python -c "
from flashlite import _cuda_naive, _cuda_tiled, _cuda_online_softmax, _cuda_fused
for name, mod in [('_cuda_naive', _cuda_naive), ('_cuda_tiled', _cuda_tiled),
                   ('_cuda_online_softmax', _cuda_online_softmax), ('_cuda_fused', _cuda_fused)]:
    print(f'flashlite.{name} imported OK:', mod.__file__)
"
