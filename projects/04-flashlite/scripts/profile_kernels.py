#!/usr/bin/env python3
"""Phase 2/3 profiler-checkpoint runner: repeatedly invokes ONE variant at
ONE fixed shape, meant to be wrapped by `nsys profile` (Nsight Systems --
the only profiler confirmed installed on this development machine; `ncu`
/ Nsight Compute is NOT installed here, the same constraint KernelForge
already documented in its own profiling/metric-guide.md and ADR 0006).
This gives a measured, profiler-captured per-kernel-name timeline (which
of compute_scores_*/softmax_rows_kernel/weighted_sum_*_kernel actually
dominates wall time), independent of and complementary to the GpuTimer-based
`scripts/run_benchmarks.py` sweep -- exactly the "measured profiler
baseline" roadmap Phase 2 asks for before any optimization claim is made.

Usage (inside the WSL venv, from the repo root, after `pip install -e .`;
requires the GPU lock per the orchestrator's hard rule, since its captured
numbers are cited in docs/io-analysis.md):

    nsys profile -o profiling/nsight-systems/<name> --force-overwrite=true -- \\
        python scripts/profile_kernels.py --variant naive --batch 1 --heads 8 \\
        --seq-len 2048 --head-dim 64 --causal --iters 30

    nsys stats profiling/nsight-systems/<name>.nsys-rep    # human-readable summary

This script itself performs NO timing/statistics of its own and emits NO
BenchResult -- it exists purely to give a profiler something representative
to attach to. `scripts/run_benchmarks.py` remains the single source of
committed BenchResult numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from flashlite.ops import attention
from flashlite.reference.tensors import DEFAULT_SEED, make_qkv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["reference", "naive", "tiled"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=30, help="timed-region iteration count")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("profile_kernels.py: no CUDA device available; aborting.", file=sys.stderr)
        return 1

    q, k, v = make_qkv(args.batch, args.heads, args.seq_len, args.head_dim, seed=args.seed, device="cuda")

    for _ in range(args.warmup):
        attention(q, k, v, causal=args.causal, variant=args.variant)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"profile_kernels:{args.variant}")
    for _ in range(args.iters):
        attention(q, k, v, causal=args.causal, variant=args.variant)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(
        f"profile_kernels.py: ran {args.variant} B={args.batch} H={args.heads} S={args.seq_len} "
        f"D={args.head_dim} causal={args.causal} for {args.iters} iterations (+{args.warmup} warmup)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
