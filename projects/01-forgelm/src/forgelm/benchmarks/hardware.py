"""Hardware/software environment fingerprinting for benchmark artifacts.

Every :class:`~forgelm.benchmarks.harness.BenchmarkResult` embeds this, so
a committed result is self-describing (methodology requirement: "every
result records hardware, software versions, dtype, seed, model config,
dataset, warmup policy, sample count" -- spec Part 3 benchmark plan).
"""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch


def hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "device_type": device.type,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_total_memory_bytes"] = int(props.total_memory)
    return info


def software_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,  # type: ignore[attr-defined]
        "platform": platform.platform(),
    }
