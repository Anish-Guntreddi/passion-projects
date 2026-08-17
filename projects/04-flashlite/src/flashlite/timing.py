"""CUDA-event-based GPU timer (mirrors KernelForge's src/common/gpu_timer.cuh).

Measures device-side elapsed time between start() and stop_ms() using
torch.cuda.Event(enable_timing=True), which wraps the same cudaEvent_t
mechanism KernelForge's GpuTimer uses directly -- the correct way to time
kernel execution, as opposed to host wall-clock timers (which include
host-side launch overhead and any host work racing ahead of the async
kernel launch). This measures *kernel* time only; H2D/D2H transfer is
outside the timed region unless explicitly included.
"""

from __future__ import annotations

import torch


class GpuTimer:
    """Usage:

        timer = GpuTimer()
        timer.start()
        <launch CUDA work>
        elapsed_ms = timer.stop_ms()   # blocks until the recorded work completes
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GpuTimer requires a CUDA device (torch.cuda.is_available() is False)")
        self._start = torch.cuda.Event(enable_timing=True)
        self._stop = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop_ms(self) -> float:
        self._stop.record()
        self._stop.synchronize()
        return self._start.elapsed_time(self._stop)
