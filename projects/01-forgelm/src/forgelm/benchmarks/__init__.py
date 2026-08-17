"""Throughput/memory/val-loss benchmark harness (Phase 5) -- kept separate
from model-correctness code (spec §1.7).

``run_benchmark`` reuses ``forgelm.training.loop.Trainer`` to execute real
optimizer steps and measures them; nothing in this package asserts
anything about the *correctness* of the model's outputs (that lives under
``tests/unit/test_model_*.py`` / ``tests/integration/test_model_overfit.py``).
No benchmark number is fabricated -- every figure that ends up in
``benchmarks/results/`` or ``benchmarks/methodology.md`` is measured
output from a real run, committed as a raw JSON artifact alongside the
hardware/software/config record :func:`forgelm.benchmarks.hardware.hardware_info`
/ :func:`~forgelm.benchmarks.hardware.software_info` capture.
"""

from forgelm.benchmarks.hardware import hardware_info, software_info
from forgelm.benchmarks.harness import BenchmarkResult, run_benchmark

__all__ = [
    "BenchmarkResult",
    "hardware_info",
    "run_benchmark",
    "software_info",
]
