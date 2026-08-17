"""Machine-readable benchmark result schema (mirrors KernelForge's
src/common/bench_result.hpp/.cpp; see docs/decisions and
benchmarks/methodology.md).

This module is the single source of truth for what every committed
FlashLite benchmark result records. The companion JSON Schema document at
benchmarks/schema/bench_result.schema.json describes the same shape for
non-Python readers/validators (scripts/validate_results.py checks every
emitted record against it). Unlike KernelForge's hand-written C++ JSON
writer (ADR 0010 there, motivated by avoiding a third-party C++ JSON
dependency), this module uses the Python standard library's `json` module
directly -- there is no analogous dependency question in Python, so no ADR
is needed for that choice.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from flashlite.env_capture import EnvironmentInfo
from flashlite.stats import TimingStats, compute_stats

SCHEMA_VERSION = "1.0"


@dataclass
class BenchResult:
    """One committed measurement of one (variant, shape) point."""

    schema_version: str = SCHEMA_VERSION
    kernel_family: str = "attention"  # always "attention" in this repo
    variant: str = ""  # "v0_reference" | "v1_naive" | ... (later phases)
    description: str = ""

    env: EnvironmentInfo = field(default_factory=EnvironmentInfo)

    # Problem shape (spec SS1.7 test matrix: shapes swept by batch/heads/
    # seq_len/head_dim/causal).
    batch: int = 0
    heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    causal: bool = False
    dtype: str = "fp32"

    warmup_iters: int = 0
    measured_reps: int = 0
    seed: int = 0

    # Every individual repetition's kernel-only elapsed time (never a single
    # timing). `finalize()` fills `stats` from this.
    raw_timings_ms: list[float] = field(default_factory=list)
    stats: TimingStats | None = None

    # "Useful" bytes moved (Q + K + V reads + O write, independent of how
    # many times a naive kernel actually re-reads K/V) -- see
    # benchmarks/methodology.md SS5 for why this is deliberately the
    # *useful* count, not the measured HW traffic: the gap between this
    # number's implied bandwidth and the measured effective bandwidth is
    # itself the Phase-2 "materialization is wasteful" evidence.
    bytes_moved: float = 0.0
    effective_bandwidth_gb_s: float = 0.0

    # 2 * (QK^T) + 2 * (PV) MACs-as-FLOPs = 4 * B*H*S^2*D.
    gflops: float = 0.0

    tolerance_atol: float = 0.0
    tolerance_rtol: float = 0.0
    correctness_passed: bool = False
    correctness_note: str = ""

    notes: str = ""


def finalize(result: BenchResult) -> None:
    """Computes result.stats from result.raw_timings_ms and
    result.effective_bandwidth_gb_s from result.bytes_moved + the resulting
    median. Call after populating raw_timings_ms/bytes_moved and before
    to_json()/append_jsonl().
    """
    result.stats = compute_stats(result.raw_timings_ms)
    if result.stats.median_ms > 0.0 and result.bytes_moved > 0.0:
        # GB/s = bytes / 1e9 / (ms / 1000) = bytes / (ms * 1e6)
        result.effective_bandwidth_gb_s = result.bytes_moved / (result.stats.median_ms * 1.0e6)
    else:
        result.effective_bandwidth_gb_s = 0.0


def to_json_dict(result: BenchResult) -> dict:
    if result.stats is None:
        raise ValueError("to_json_dict: call finalize(result) before serializing (stats is None)")

    d = asdict(result)
    # Rename dataclass field names to the schema's field names.
    d["stats_ms"] = d.pop("stats")
    return d


def to_json(result: BenchResult) -> str:
    return json.dumps(to_json_dict(result), sort_keys=False)


def append_jsonl(path: str, result: BenchResult) -> None:
    """Appends one JSON line to `path` (creates the parent directory if
    needed). Never truncates -- results accumulate across runs.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(to_json(result) + "\n")
