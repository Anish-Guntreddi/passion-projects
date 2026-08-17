#!/usr/bin/env python3
"""Validates every line of every benchmarks/raw/*.jsonl file against
benchmarks/schema/bench_result.schema.json (ADR 0010). Run by
scripts/test.sh after every test run.

Usage:
    python3 scripts/validate_results.py benchmarks/raw benchmarks/schema/bench_result.schema.json
"""
import json
import pathlib
import sys

try:
    import jsonschema
except ImportError:
    # NOTE: this must NOT be sys.exit(0). scripts/test.sh runs this script as
    # the last step of the test suite specifically to enforce the FR4/ADR
    # 0010 benchmark contract on every committed benchmarks/raw/*.jsonl
    # record. Exiting 0 here would make "jsonschema isn't installed" silently
    # indistinguishable from "every record validated", letting bad/malformed
    # committed benchmark evidence pass CI on any machine that lacks the
    # package (tests/test_bench_schema.cpp only covers the C++ writer's own
    # structural output, not hand-edited or historically-committed jsonl).
    print(
        "validate_results: FATAL: the 'jsonschema' package is required to validate "
        "committed benchmark results against benchmarks/schema/bench_result.schema.json "
        "(FR4, ADR 0010) and is not installed. Schema validation cannot be silently "
        "skipped -- install it and re-run:\n"
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
        print(f"validate_results: no *.jsonl files under {raw_dir} yet -- nothing to validate.")
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

            # Cross-field invariant the JSON Schema itself cannot express:
            # Draft-07 (and the `jsonschema` package's Draft7Validator) has no
            # standard keyword to compare one property's array length against
            # a sibling property's numeric value -- that needs a non-standard
            # `$data` reference extension this validator doesn't implement.
            # Enforced here instead so a record cannot claim N measured_reps
            # while committing fewer (or more) raw_timings_ms entries, which
            # would silently weaken the FR4 "median + distribution" evidence
            # every committed result is supposed to carry. See
            # benchmarks/schema/bench_result.schema.json's raw_timings_ms
            # description, which documents this same invariant.
            reps = record.get("measured_reps")
            timings = record.get("raw_timings_ms")
            if isinstance(reps, int) and isinstance(timings, list) and len(timings) != reps:
                print(
                    f"{path}:{lineno}: SCHEMA VIOLATION: raw_timings_ms has "
                    f"{len(timings)} entries but measured_reps={reps} "
                    "(FR4: raw_timings_ms length must equal measured_reps)",
                    file=sys.stderr,
                )
                errors += 1

    if errors:
        print(f"validate_results: {errors} violation(s) across {total} record(s) in "
              f"{len(jsonl_files)} file(s).", file=sys.stderr)
        return 1

    print(f"validate_results: {total} record(s) across {len(jsonl_files)} file(s) all valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
