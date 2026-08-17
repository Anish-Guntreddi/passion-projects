"""Machine-readable environment capture (mirrors KernelForge's
src/common/bench_result.hpp:EnvironmentInfo::capture -- GPU model + compute
capability, CUDA driver/runtime version, framework version, a locked-clocks
flag + explanatory note, an observed (not locked) SM/memory clock snapshot,
an OS note, and a UTC timestamp), adapted for a PyTorch/pybind11 harness
(no separate host/CUDA *compiler* identity to report the way KernelForge's
CMake build does -- torch.utils.cpp_extension records that at build time,
not at benchmark-capture time; see docs/decisions/0005 for the build route).
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

# ADR: same WSL2-guest limitation KernelForge already documented and
# verified (docs/decisions/0005-benchmark-noise-control.md in
# ../02-kernelforge). Recorded per-result here too so no individual
# FlashLite result can be misread as having controlled clocks.
_CLOCK_LOCK_NOTE = (
    "GPU clocks are NOT locked. `nvidia-smi -lgc ...` returns 'The current "
    "user does not have permission to change clocks' under this WSL2 guest "
    "(same constraint documented and verified in "
    "../02-kernelforge/docs/decisions/0005-benchmark-noise-control.md). "
    "observed_sm_clock_mhz/observed_mem_clock_mhz below are read-only "
    "snapshots taken at result-capture time, not a controlled state."
)


@dataclass
class EnvironmentInfo:
    gpu_name: str = ""
    compute_capability: str = ""  # e.g. "8.9"
    cuda_driver_version: str = ""
    cuda_runtime_version: str = ""  # torch.version.cuda
    python_version: str = ""
    torch_version: str = ""
    build_route_note: str = "pybind11 + torch.utils.cpp_extension (ADR 0005); nvcc invoked by torch at build time."
    locked_clocks: bool = False
    clock_lock_note: str = ""
    observed_sm_clock_mhz: int = -1
    observed_mem_clock_mhz: int = -1
    os_note: str = "Linux (WSL2 Ubuntu) guest on Windows 11 host"
    timestamp_utc: str = ""

    @staticmethod
    def capture() -> EnvironmentInfo:
        env = EnvironmentInfo()
        env.python_version = platform.python_version()
        env.torch_version = torch.__version__
        env.cuda_runtime_version = torch.version.cuda or ""
        env.timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        env.locked_clocks = False
        env.clock_lock_note = _CLOCK_LOCK_NOTE

        if torch.cuda.is_available():
            env.gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            env.compute_capability = f"{major}.{minor}"

        sm_clock, mem_clock, driver_version = _read_nvidia_smi()
        env.observed_sm_clock_mhz = sm_clock
        env.observed_mem_clock_mhz = mem_clock
        env.cuda_driver_version = driver_version

        return env


def _read_nvidia_smi() -> tuple[int, int, str]:
    """Best-effort nvidia-smi query for observed clocks + driver version.

    Returns (-1, -1, "") if nvidia-smi is unavailable or output is
    unparseable -- this is an observability nicety, not load-bearing, so it
    fails soft rather than raising (same contract as KernelForge's
    read_observed_clocks_mhz()).
    """
    for smi_path in ("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"):
        try:
            result = subprocess.run(
                [smi_path, "--query-gpu=clocks.sm,clocks.mem,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            sm_clock = int(parts[0])
            mem_clock = int(parts[1])
            driver_version = parts[2]
            return sm_clock, mem_clock, driver_version
        except Exception:  # noqa: BLE001, S112 -- intentionally broad: nvidia-smi can
            # fail in many shapes (missing binary, timeout, unexpected output
            # format); any of them should fall through to the next candidate
            # path / the soft (-1, -1, "") default below, not raise, per this
            # function's documented best-effort contract.
            continue
    return -1, -1, ""
