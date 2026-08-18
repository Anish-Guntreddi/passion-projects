#!/usr/bin/env python3
"""Validates every line of every profiling/occupancy/*.jsonl file against
profiling/schema/occupancy_report.schema.json (ADR 0014). Companion to
scripts/validate_results.py (ADR 0010) for the Phase 6 occupancy-evidence
records instead of benchmarks/raw/ timing records.

Usage:
    python3 scripts/validate_occupancy.py profiling/occupancy profiling/schema/occupancy_report.schema.json
"""
import json
import pathlib
import sys

try:
    import jsonschema
except ImportError:
    # See scripts/validate_results.py's identical comment: this must NOT be
    # sys.exit(0) -- silently skipping validation on a machine without
    # jsonschema would let a malformed committed occupancy record pass.
    print(
        "validate_occupancy: FATAL: the 'jsonschema' package is required to validate "
        "committed profiling/occupancy/*.jsonl records against "
        "profiling/schema/occupancy_report.schema.json (ADR 0014) and is not installed. "
        "Install it and re-run:\n"
        "    python3 -m pip install --user jsonschema",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <raw-dir> <schema-file>", file=sys.stderr)
        return 2

    raw_dir = pathlib.Path(sys.argv[1])
    schema_path = pathlib.Path(sys.argv[2])
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft7Validator(schema)

    jsonl_files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.is_dir() else []
    if not jsonl_files:
        print(f"validate_occupancy: no *.jsonl files under {raw_dir} yet -- nothing to validate.")
        return 0

    total = 0
    errors = 0
    for path in jsonl_files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{lineno}: INVALID JSON: {e}", file=sys.stderr)
                errors += 1
                continue
            for error in validator.iter_errors(record):
                print(f"{path}:{lineno}: SCHEMA VIOLATION: {error.message}", file=sys.stderr)
                errors += 1

    if errors:
        print(f"validate_occupancy: {errors} violation(s) across {total} record(s) in "
              f"{len(jsonl_files)} file(s).", file=sys.stderr)
        return 1

    print(f"validate_occupancy: {total} record(s) across {len(jsonl_files)} file(s) all valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
