"""Phase 0 exit criterion: CPU smoke test (spec section 1.8)."""

from __future__ import annotations

import json
import subprocess
import sys

from forgelm.cli.smoke import run_smoke
from forgelm.config import SmokeConfig


def test_smoke_cpu_is_deterministic_across_calls() -> None:
    config = SmokeConfig(seed=1234, size=32, device="cpu")
    result_1 = run_smoke(config)
    result_2 = run_smoke(config)
    assert result_1["device"] == "cpu"
    assert result_1["cuda_available"] == result_2["cuda_available"]
    # Numerical tolerance, not exact equality, per project-wide test policy --
    # though CPU float ops are in practice bit-identical for a fixed op
    # sequence and seed.
    assert abs(result_1["checksum"] - result_2["checksum"]) < 1e-9


def test_smoke_cpu_different_seeds_produce_different_output() -> None:
    result_1 = run_smoke(SmokeConfig(seed=1, size=16, device="cpu"))
    result_2 = run_smoke(SmokeConfig(seed=2, size=16, device="cpu"))
    assert result_1["checksum"] != result_2["checksum"]


def test_smoke_cli_end_to_end_is_deterministic() -> None:
    """Invoke the actual `forgelm smoke` CLI command as a subprocess, twice."""
    cmd = [
        sys.executable,
        "-m",
        "forgelm.cli.main",
        "smoke",
        "--seed",
        "7",
        "--size",
        "16",
        "--device",
        "cpu",
    ]
    out_1 = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
    out_2 = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
    data_1 = json.loads(out_1)
    data_2 = json.loads(out_2)
    assert data_1["seed"] == data_2["seed"] == 7
    assert abs(data_1["checksum"] - data_2["checksum"]) < 1e-9
