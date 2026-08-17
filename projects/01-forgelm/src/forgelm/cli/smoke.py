"""Phase 0 placeholder end-to-end command.

Exit criterion: "placeholder end-to-end command runs deterministically."
:func:`run_smoke` seeds every RNG, does a real (if tiny) tensor computation
on the requested device, and returns a checksum. Two invocations with the
same :class:`~forgelm.config.SmokeConfig` must agree to within the numeric
tolerance used everywhere else in this project (never exact float
equality, per the project-wide numerical-testing rule).
"""

from __future__ import annotations

from typing import Any

import torch

from forgelm.config import SmokeConfig
from forgelm.seed import set_seed


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Run the deterministic placeholder pipeline and return a result record."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SmokeConfig.device='cuda' but no CUDA device is available")

    set_seed(config.seed)
    device = torch.device(config.device)

    # A stand-in for "the pipeline": seed -> deterministic random init ->
    # a couple of ops that resemble the matmuls the real model will do.
    x = torch.randn(config.size, config.size, device=device)
    w = torch.randn(config.size, config.size, device=device)
    y = torch.tanh(x @ w)
    checksum = float(y.sum().item())

    return {
        "seed": config.seed,
        "size": config.size,
        "device": config.device,
        "checksum": checksum,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
