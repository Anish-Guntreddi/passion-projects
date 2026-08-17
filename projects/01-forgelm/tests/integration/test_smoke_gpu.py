"""Phase 0 exit criterion: GPU smoke test, run when hardware/CI permits (spec 1.8).

Automatically skipped (not failed) when no CUDA device is visible, so this
suite is safe to run on CPU-only machines too.
"""

from __future__ import annotations

import pytest
import torch

from forgelm.cli.smoke import run_smoke
from forgelm.config import SmokeConfig

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


@requires_cuda
def test_smoke_gpu_is_deterministic_across_calls() -> None:
    config = SmokeConfig(seed=99, size=64, device="cuda")
    result_1 = run_smoke(config)
    result_2 = run_smoke(config)
    assert result_1["device"] == "cuda"
    assert result_1["cuda_available"] is True
    # cuBLAS reduction order is not guaranteed bit-identical run to run even
    # with deterministic algorithms requested, so use a tolerance rather
    # than exact float equality (project-wide numerical test policy).
    scale = max(1.0, abs(result_1["checksum"]))
    assert abs(result_1["checksum"] - result_2["checksum"]) < 1e-3 * scale


@requires_cuda
def test_smoke_gpu_runs_on_the_visible_device() -> None:
    result = run_smoke(SmokeConfig(seed=1, size=32, device="cuda"))
    assert result["device"] == "cuda"
    assert torch.cuda.get_device_name(0)  # just needs to not raise


def test_smoke_config_rejects_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        run_smoke(SmokeConfig(seed=1, size=8, device="cuda"))
