"""Shared pytest configuration: auto-skips every test marked `gpu` with a
specific, honest reason (no CUDA device present, vs. CUDA present but one
of the compiled extensions hasn't been built yet) instead of letting it
fail with an ImportError or a CUDA runtime error.

Both compiled extensions (`_cuda_naive` V1, `_cuda_tiled` V2) are built
together by one `scripts/build_ext.sh` run (setup.py lists both as
ext_modules of the same `pip install -e .`), so in this repo's actual
supported workflow they are always available together or missing
together -- a single combined availability flag is accurate, not an
approximation, and keeps every `gpu`-marked test (whichever extension it
happens to exercise) skipping with the same clear reason in a fresh clone
that hasn't run the build step yet.
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

_MISSING_EXTENSIONS: list[str] = []
try:
    from flashlite import _cuda_naive  # noqa: F401
except ImportError:
    _MISSING_EXTENSIONS.append("flashlite._cuda_naive")

try:
    from flashlite import _cuda_tiled  # noqa: F401
except ImportError:
    _MISSING_EXTENSIONS.append("flashlite._cuda_tiled")

CUDA_NAIVE_EXT_AVAILABLE = "flashlite._cuda_naive" not in _MISSING_EXTENSIONS
CUDA_TILED_EXT_AVAILABLE = "flashlite._cuda_tiled" not in _MISSING_EXTENSIONS
CUDA_EXTENSIONS_AVAILABLE = not _MISSING_EXTENSIONS


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_no_cuda = pytest.mark.skip(reason="requires a CUDA device (torch.cuda.is_available() is False)")
    skip_no_ext = pytest.mark.skip(
        reason=f"extension(s) not built: {', '.join(_MISSING_EXTENSIONS)}; run `pip install -e .` first"
    )
    for item in items:
        if "gpu" not in item.keywords:
            continue
        if not CUDA_AVAILABLE:
            item.add_marker(skip_no_cuda)
        elif not CUDA_EXTENSIONS_AVAILABLE:
            item.add_marker(skip_no_ext)
