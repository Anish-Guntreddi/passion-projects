# ADR 0010: Hand-Rolled JSON for the Benchmark Result Schema

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
FR3/FR4 require a "machine-readable benchmark result schema" capturing
GPU model + compute capability, CUDA/toolkit/compiler versions, clock
caveats, input dimensions, block/grid config, warmup count, measured
repetitions, median + distribution, effective bandwidth/FLOP/s where
meaningful, correctness tolerance, and compiler flags. The roadmap
requires this schema and `benchmarks/methodology.md` to exist **before**
any headline benchmark numbers are collected (Phase 0 exit criterion:
"benchmark output machine-readable").

## Decision
1. Define the schema as a plain C++ struct,
   `kernelforge::BenchResult` (`src/common/bench_result.hpp`), and a
   matching **JSON Schema document**,
   `benchmarks/schema/bench_result.schema.json`, committed before any
   `benchmarks/raw/*.jsonl` file is written.
2. Serialize with a small hand-written JSON writer
   (`kernelforge::to_json`, in `bench_result.cpp`) rather than a third-
   party JSON library. The schema is flat enough (one object, a handful
   of nested sub-objects, one array of doubles) that a hand-written
   writer is ~80 lines, has zero dependencies, and is easy to audit for
   correct escaping — appropriate for a project that values being
   interview-explainable end-to-end (hard constraint 6) over pulling in
   `nlohmann/json` for a narrow, fixed shape.
3. Every benchmark binary appends one JSON object per line (JSON Lines,
   `.jsonl`) to a file under `benchmarks/raw/`, one file per kernel
   family, so results accumulate across runs/dates without clobbering
   history and remain trivially parseable by `python3 -c
   "import json; [json.loads(l) for l in open(f)]"` or pandas
   (`pd.read_json(path, lines=True)`) in the Phase 7 analysis pipeline
   (FR7) without any custom parser.
4. `python3`'s built-in `jsonschema` package (confirmed present on the
   dev machine) can validate `benchmarks/raw/*.jsonl` lines against
   `benchmarks/schema/bench_result.schema.json` — `scripts/validate_results.py`
   does this and is run as part of `scripts/test.sh`.

## Consequences
- No new C++ dependency. Schema evolution is a two-file change (struct +
  JSON Schema); `schema_version` is stamped into every record
  (`"1.0"` for Phases 0–1) so future incompatible changes are detectable
  by readers instead of silently misparsed.
- The JSON Schema doc is also the most concrete, unambiguous statement of
  the FR4 benchmark contract — anyone reviewing this repo can read one
  file to see exactly what every committed result is required to record.
