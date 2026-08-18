#!/usr/bin/env python3
"""Benchmark driver (mirrors KernelForge's scripts/bench_driver.py in
spirit -- config-driven, stdlib-only orchestration -- but runs in-process
rather than spawning a separate binary per point, since here "the binary"
IS the Python process importing flashlite directly).

Reads a sweep spec from benchmarks/configs/attention.json, runs each point
(correctness-gated: a point whose output does not match the V0 reference
within tolerance is skipped with a printed error, never emitted as a
BenchResult -- spec hard constraint: never suppress a numerical mismatch to
improve benchmark appearance), and appends one validated BenchResult per
point to benchmarks/raw/attention.jsonl.

Usage (inside the WSL venv, from the repo root, after `pip install -e .`):
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --config benchmarks/configs/attention.json --out benchmarks/raw/attention.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from flashlite.bench_schema import BenchResult, append_jsonl, finalize
from flashlite.compare import DEFAULT_ATOL, DEFAULT_RTOL, allclose_compare
from flashlite.env_capture import EnvironmentInfo
from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import make_qkv
from flashlite.timing import GpuTimer

# ADR 0007: kAttnTileDim, mirrored here because it is a C++ constexpr
# inside the compiled _cuda_tiled extension, not a Python-readable value
# (ADR 0008 explains why this benchmark driver hard-codes it rather than
# importing it). ADR 0009: "online_softmax" reuses the same constant for
# kernel 1/3 (unchanged from "tiled"). ADR 0011: "fused" has its own
# kFusedTileDim, currently also 32.
TILE_SIZE_V2 = 32
TILE_SIZE_V3 = 32
TILE_SIZE_V4 = 32

VARIANT_TO_SCHEMA_NAME = {
    "reference": "v0_reference",
    "naive": "v1_naive",
    "tiled": "v2_tiled",
    "online_softmax": "v3_online_softmax",
    "fused": "v4_fused",
}
VARIANT_TO_TILE_SIZE = {
    "reference": 0,
    "naive": 0,
    "tiled": TILE_SIZE_V2,
    "online_softmax": TILE_SIZE_V3,
    "fused": TILE_SIZE_V4,
}


def _run_variant(variant: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    if variant == "reference":
        return reference_attention(q, k, v, causal=causal)
    if variant == "naive":
        from flashlite import _cuda_naive

        return _cuda_naive.attention_naive_forward(q, k, v, causal)
    if variant == "tiled":
        from flashlite import _cuda_tiled

        return _cuda_tiled.attention_tiled_forward(q, k, v, causal)
    if variant == "online_softmax":
        from flashlite import _cuda_online_softmax

        return _cuda_online_softmax.attention_online_softmax_forward(q, k, v, causal)
    if variant == "fused":
        from flashlite import _cuda_fused

        return _cuda_fused.attention_fused_forward(q, k, v, causal)
    raise ValueError(f"unknown variant: {variant!r}")


def _measure_peak_memory_bytes(variant: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> float:
    """ADR 0010: one additional, untimed forward call, deliberately kept
    separate from the raw_timings_ms measurement loop below (repeated timed
    calls may reuse already-allocated caching-allocator memory from earlier
    reps, which would not reflect a single call's true peak).
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _run_variant(variant, q, k, v, causal)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated())


def run_point(
    point: dict,
    seed: int,
    warmup_iters: int,
    measured_reps: int,
    env: EnvironmentInfo,
    measure_peak_memory: bool = False,
) -> BenchResult | None:
    variant = point["variant"]
    batch, heads, seq_len, head_dim = point["batch"], point["heads"], point["seq_len"], point["head_dim"]
    causal = point["causal"]

    q, k, v = make_qkv(batch, heads, seq_len, head_dim, seed=seed, device="cuda")

    expected = reference_attention(q, k, v, causal=causal)
    actual = _run_variant(variant, q, k, v, causal)
    torch.cuda.synchronize()

    cmp_result = allclose_compare(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    if not cmp_result.passed:
        print(f"  SKIPPED (correctness FAILED): {variant} B={batch} H={heads} S={seq_len} D={head_dim} "
              f"causal={causal}: {cmp_result.summary()}", file=sys.stderr)
        return None

    peak_memory_bytes = 0.0
    if measure_peak_memory:
        peak_memory_bytes = _measure_peak_memory_bytes(variant, q, k, v, causal)

    timer = GpuTimer()
    for _ in range(warmup_iters):
        _run_variant(variant, q, k, v, causal)
    torch.cuda.synchronize()

    raw_timings_ms = []
    for _ in range(measured_reps):
        timer.start()
        _run_variant(variant, q, k, v, causal)
        raw_timings_ms.append(timer.stop_ms())

    bytes_moved = 4.0 * (4 * batch * heads * seq_len * head_dim)  # Q+K+V read, O write, fp32
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim  # QK^T + PV, 2 flops/MAC each

    result = BenchResult(
        variant=VARIANT_TO_SCHEMA_NAME[variant],
        description=f"{variant} attention forward, B={batch} H={heads} S={seq_len} D={head_dim} causal={causal}",
        env=env,
        batch=batch, heads=heads, seq_len=seq_len, head_dim=head_dim, causal=causal,
        dtype="fp32",
        warmup_iters=warmup_iters, measured_reps=measured_reps, seed=seed,
        raw_timings_ms=raw_timings_ms,
        bytes_moved=bytes_moved,
        gflops=0.0,  # filled in below once stats.median_ms is known
        tolerance_atol=DEFAULT_ATOL, tolerance_rtol=DEFAULT_RTOL,
        correctness_passed=True,
        correctness_note=cmp_result.summary(),
        tile_size=VARIANT_TO_TILE_SIZE[variant],  # ADR 0008/0009/0011
        peak_memory_bytes=peak_memory_bytes,  # ADR 0010
    )
    finalize(result)
    if result.stats and result.stats.median_ms > 0:
        result.gflops = flops / (result.stats.median_ms * 1.0e6)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmarks" / "configs" / "attention.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "raw" / "attention.jsonl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("run_benchmarks.py: no CUDA device available; aborting.", file=sys.stderr)
        return 1

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    env = EnvironmentInfo.capture()

    measure_peak_memory = bool(config.get("measure_peak_memory", False))  # ADR 0010, opt-in per config

    emitted = 0
    for point in config["points"]:
        result = run_point(
            point,
            seed=config["seed"],
            warmup_iters=config["warmup_iters"],
            measured_reps=config["measured_reps"],
            env=env,
            measure_peak_memory=measure_peak_memory,
        )
        if result is None:
            continue
        append_jsonl(args.out, result)
        emitted += 1
        peak_mb_note = f", peak={result.peak_memory_bytes / 1.0e6:.2f} MB" if result.peak_memory_bytes > 0 else ""
        print(f"  OK: {result.variant} B={result.batch} H={result.heads} S={result.seq_len} "
              f"D={result.head_dim} causal={result.causal} -> median={result.stats.median_ms:.4f} ms, "
              f"{result.effective_bandwidth_gb_s:.2f} GB/s, {result.gflops:.2f} GFLOP/s{peak_mb_note}")

    print(f"run_benchmarks.py: emitted {emitted}/{len(config['points'])} points to {args.out}")
    return 0 if emitted == len(config["points"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
