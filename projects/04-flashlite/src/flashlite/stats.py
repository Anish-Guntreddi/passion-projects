"""Distribution statistics over raw timing samples (mirrors KernelForge's
src/common/stats.hpp/.cpp exactly, including the linear-interpolation
percentile method -- numpy's default "linear" method -- so results are
directly comparable across the two projects' benchmark JSON).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TimingStats:
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stddev_ms: float
    p25_ms: float
    p75_ms: float


def _percentile(sorted_samples: list[float], p: float) -> float:
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    idx = p * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    frac = idx - lo
    if lo == hi:
        return sorted_samples[lo]
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])


def compute_stats(samples_ms: list[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("compute_stats: samples_ms must be non-empty")

    s = sorted(samples_ms)
    n = len(s)
    mean = sum(s) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in s) / (n - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    return TimingStats(
        min_ms=s[0],
        max_ms=s[-1],
        mean_ms=mean,
        median_ms=_percentile(s, 0.5),
        stddev_ms=stddev,
        p25_ms=_percentile(s, 0.25),
        p75_ms=_percentile(s, 0.75),
    )
