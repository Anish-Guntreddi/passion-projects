#!/usr/bin/env python3
"""Runs a benchmarks/configs/*.json sweep against a compiled bench_* binary
and appends each resulting BenchResult JSON line to benchmarks/raw/.

The C++ binary is the single source of truth for HOW a measurement is
taken (warmup, CUDA-event timing, stats, correctness gating); this script
only supplies WHAT to measure (which sizes/variants), read verbatim from
the committed config so nothing about a sweep is hand-typed twice. Stdlib
only (json, subprocess, argparse) -- runs on a fresh clone with no pip
install (ADR 0009's same rationale, applied to the analysis-adjacent
tooling).

Usage:
    python3 scripts/bench_driver.py \
        --config benchmarks/configs/vector_add.json \
        --bin-dir build/apps \
        --out benchmarks/raw/vector_add.jsonl
"""
import argparse
import json
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="benchmarks/configs/*.json sweep spec")
    parser.add_argument("--bin-dir", required=True, help="directory containing the built bench_* binaries")
    parser.add_argument("--out", required=True, help="benchmarks/raw/<family>.jsonl to append to")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    cfg = json.loads(config_path.read_text())

    binary = pathlib.Path(args.bin_dir) / cfg["binary"]
    if not binary.exists():
        print(f"error: binary not found: {binary} (did you run scripts/build.sh?)", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants = cfg["variants"]
    sweep_param = cfg["sweep_param"]
    sweep_values = cfg["sweep_values"]
    warmup = cfg.get("warmup_iters", 10)
    reps = cfg.get("measured_reps", 30)
    seed = cfg.get("seed", 123456789)
    fixed_args = cfg.get("fixed_args", {})

    lines_written = 0
    with out_path.open("a") as out_f:
        for variant in variants:
            for value in sweep_values:
                cmd = [
                    str(binary),
                    "--variant", str(variant),
                    f"--{sweep_param}", str(value),
                    "--warmup", str(warmup),
                    "--reps", str(reps),
                    "--seed", str(seed),
                ]
                for k, v in fixed_args.items():
                    cmd += [f"--{k}", str(v)]

                print("+", " ".join(cmd), file=sys.stderr)
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                # The binary always prints diagnostics to stderr, so surface
                # them regardless of outcome -- useful for warmup/median
                # sanity-checking while a sweep is running.
                if result.stderr:
                    print(result.stderr.rstrip(), file=sys.stderr)

                if result.returncode != 0:
                    print(result.stdout, file=sys.stderr)
                    raise SystemExit(
                        f"bench_driver: '{cmd[0]}' exited {result.returncode} for "
                        f"variant={variant} {sweep_param}={value} -- refusing to write a "
                        f"partial/failed result. See stderr above."
                    )

                stdout_lines = [l for l in result.stdout.splitlines() if l.strip()]
                if not stdout_lines:
                    raise SystemExit(
                        f"bench_driver: '{cmd[0]}' produced no stdout for "
                        f"variant={variant} {sweep_param}={value}"
                    )
                line = stdout_lines[-1]
                parsed = json.loads(line)  # validate shape before committing to disk
                if not parsed.get("correctness_passed", False):
                    raise SystemExit(
                        f"bench_driver: refusing to write a result with "
                        f"correctness_passed=false (variant={variant} {sweep_param}={value})"
                    )

                out_f.write(line + "\n")
                out_f.flush()
                lines_written += 1

    print(f"wrote {lines_written} result(s) to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
