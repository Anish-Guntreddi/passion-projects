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
    parser.add_argument(
        "--only-variant",
        action="append",
        default=None,
        help="restrict this run to one of the config's variants (repeatable). Used for "
        "backfilling a single variant's numbers (e.g. a rung added to a ladder after the "
        "rest was already benchmarked) without re-running -- and re-appending duplicate "
        "records for -- every other variant already committed to the same .jsonl.",
    )
    parser.add_argument(
        "--only-sweep-value",
        action="append",
        default=None,
        help="restrict this run to one of the config's sweep_values (repeatable, matched by "
        "str(value)). Used the same way as --only-variant, but for excluding a specific "
        "swept size -- e.g. one that genuinely fails a variant's correctness gate at that "
        "scale (never committed, per house rule) while every other configured size for "
        "that variant is still collected with the same fixed methodology.",
    )
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
    if args.only_variant:
        unknown = [v for v in args.only_variant if v not in variants]
        if unknown:
            print(
                f"error: --only-variant {unknown} not present in {config_path}'s "
                f"variants list {variants}",
                file=sys.stderr,
            )
            return 1
        variants = [v for v in variants if v in args.only_variant]
    sweep_param = cfg["sweep_param"]
    sweep_values = cfg["sweep_values"]
    if args.only_sweep_value:
        wanted = set(args.only_sweep_value)
        unknown = [v for v in wanted if str(v) not in {str(sv) for sv in sweep_values}]
        if unknown:
            print(
                f"error: --only-sweep-value {unknown} not present in {config_path}'s "
                f"sweep_values list {sweep_values}",
                file=sys.stderr,
            )
            return 1
        sweep_values = [sv for sv in sweep_values if str(sv) in wanted]
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
