#!/usr/bin/env python3
"""Validates every line of every benchmarks/raw/*.jsonl file against
benchmarks/schema/bench_result.schema.json (mirrors KernelForge's
scripts/validate_results.py), plus one cross-field check draft-07 JSON
Schema cannot express: len(raw_timings_ms) == measured_reps.

Exits non-zero (with every failure printed) if any record is invalid, or
if no benchmarks/raw/*.jsonl files exist yet (nothing to validate is a
separate, quieter no-op exit 0 case controlled by --allow-empty).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "schema" / "bench_result.schema.json"
RAW_DIR = REPO_ROOT / "benchmarks" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", action="store_true",
                         help="exit 0 even if no benchmarks/raw/*.jsonl files exist yet")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)

    jsonl_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not jsonl_files:
        if args.allow_empty:
            print("validate_results.py: no benchmarks/raw/*.jsonl files yet (--allow-empty set); OK.")
            return 0
        print("validate_results.py: no benchmarks/raw/*.jsonl files found.", file=sys.stderr)
        return 1

    total_records = 0
    total_errors = 0
    for path in jsonl_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{lineno}: invalid JSON: {e}", file=sys.stderr)
                total_errors += 1
                continue

            errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
            for err in errors:
                print(f"{path}:{lineno}: schema violation at {list(err.path)}: {err.message}", file=sys.stderr)
                total_errors += 1

            raw_timings = record.get("raw_timings_ms", [])
            measured_reps = record.get("measured_reps", -1)
            if len(raw_timings) != measured_reps:
                print(
                    f"{path}:{lineno}: len(raw_timings_ms)={len(raw_timings)} != "
                    f"measured_reps={measured_reps}",
                    file=sys.stderr,
                )
                total_errors += 1

    if total_errors:
        print(f"validate_results.py: {total_errors} error(s) across {total_records} record(s) in "
              f"{len(jsonl_files)} file(s).", file=sys.stderr)
        return 1

    print(f"validate_results.py: {total_records} record(s) across {len(jsonl_files)} file(s) all valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
