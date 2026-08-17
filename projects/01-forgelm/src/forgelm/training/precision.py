"""Mixed-precision policy (D6): bf16 autocast on CUDA when supported,
plain fp32 otherwise.

See ``docs/decisions/0008-precision-strategy.md``. bf16 (not fp16) is
used when autocasting because it has the same exponent range as fp32 --
no loss-scaling machinery (``GradScaler``) is needed to avoid gradient
underflow, which keeps the training loop simpler for a project whose
first priority is correctness/explainability, not maximum throughput.
"""

from __future__ import annotations

import dataclasses

import torch

_VALID_REQUESTS = ("auto", "fp32", "bf16")


@dataclasses.dataclass(frozen=True)
class PrecisionPolicy:
    """What :class:`~forgelm.training.loop.Trainer` passes to ``torch.autocast``."""

    device_type: str
    autocast_dtype: torch.dtype | None  # None = no autocast, plain fp32 compute

    @property
    def enabled(self) -> bool:
        return self.autocast_dtype is not None


def resolve_precision(device: torch.device, requested: str = "auto") -> PrecisionPolicy:
    """Resolve a :class:`PrecisionPolicy` for ``device`` given ``requested``.

    ``requested`` is one of:
      - ``"auto"`` (default): D6's recommended default -- bf16 autocast if
        ``device`` is CUDA and the GPU supports bf16, else plain fp32.
      - ``"fp32"``: always plain fp32, regardless of device.
      - ``"bf16"``: force bf16 autocast; raises if ``device`` is not CUDA
        (there is no CPU bf16-autocast policy in this project -- D3's "CPU
        path must work for tests" is satisfied by fp32 CPU, not by CPU
        autocast, which this project does not exercise).
    """
    if requested not in _VALID_REQUESTS:
        raise ValueError(f"requested precision must be one of {_VALID_REQUESTS}, got {requested!r}")

    if requested == "fp32":
        return PrecisionPolicy(device.type, None)

    if requested == "bf16":
        if device.type != "cuda":
            raise ValueError(
                f"bf16 precision requires a CUDA device, got device.type={device.type!r}"
            )
        return PrecisionPolicy(device.type, torch.bfloat16)

    # "auto"
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return PrecisionPolicy(device.type, torch.bfloat16)
    return PrecisionPolicy(device.type, None)
