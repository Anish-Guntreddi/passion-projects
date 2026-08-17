"""Shared pytest configuration: auto-skips every test marked `gpu` with a
specific, honest reason (no CUDA device present, vs. CUDA present but the
V1 extension hasn't been built yet) instead of letting it fail with an
ImportError or a CUDA runtime error.
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    from flashlite import _cuda_naive  # noqa: F401

    CUDA_NAIVE_EXT_AVAILABLE = True
except ImportError:
    CUDA_NAIVE_EXT_AVAILABLE = False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_no_cuda = pytest.mark.skip(reason="requires a CUDA device (torch.cuda.is_available() is False)")
    skip_no_ext = pytest.mark.skip(
        reason="flashlite._cuda_naive extension not built; run `pip install -e .` first"
    )
    for item in items:
        if "gpu" not in item.keywords:
            continue
        if not CUDA_AVAILABLE:
            item.add_marker(skip_no_cuda)
        elif not CUDA_NAIVE_EXT_AVAILABLE:
            item.add_marker(skip_no_ext)
