# ADR 0008: BenchResult Schema Extension for Phase 3 (`tile_size` field)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Phase 3 introduces `v2_tiled`, the first variant in
  this repo whose kernel launch config has a tunable tile-size constant
  (ADR 0007). Phase 6 (roadmap: "Tile-size/block-shape sweep ... tuned
  defaults based on benchmark evidence, not folklore") needs every V2
  `BenchResult` to say what tile size actually produced it, or that later
  sweep has nothing concrete to compare against.

## Context
This mirrors KernelForge's own ADR 0011 ("BenchResult Schema Extension for
Phase 2/3") almost exactly: a new variant's benchmark needs one piece of
per-result metadata no earlier variant required (`num_bins`/
`contention_profile` there, for histogram; `tile_size` here, for the tiled
kernel launch config), and the schema needs to grow additively without
breaking any already-committed record.

## Decision
1. Extend `flashlite.bench_schema.BenchResult` with one new field:
   `tile_size: int = 0`. `0` means "not applicable" (every `v0_reference`
   and `v1_naive` record, neither of which has a tile-size concept) --
   mirrors KernelForge ADR 0011's `num_bins`/`contention_profile` "not
   applicable" sentinel convention exactly, rather than inventing a new
   null-handling convention for this repo.
2. `scripts/run_benchmarks.py` populates `tile_size=32`
   (`flashlite._cuda_tiled.attention_tiled.kAttnTileDim`, ADR 0007) for
   every `variant="tiled"` point, and leaves it at the `0` default for
   `"reference"`/`"naive"` points -- there is no Python-side constant to
   import (the tile size is a C++ `constexpr` inside the compiled
   extension, not exposed as a Python-readable value), so the benchmark
   driver hard-codes `32` next to a comment pointing at ADR 0007/the
   `.cuh` constant as the source of truth, the same way it already
   hard-codes the `bytes_moved`/`gflops` formulas next to a comment citing
   `docs/attention-math.md`.
3. Add `tile_size` to `benchmarks/schema/bench_result.schema.json` as an
   **optional** property (`"type": "integer", "minimum": 0`), not added to
   `required` -- every existing Phase 0/1 record (which predates this
   field entirely) remains schema-valid unchanged, exactly as KernelForge's
   ADR 0011 point 3 does for its own additive fields.
4. `schema_version` stays `"1.0"`, for the identical reasoning as
   KernelForge ADR 0011 point 4: this is an additive, backward-compatible
   change (one new optional field, no removed/renamed/redefined field), not
   the kind of incompatible change `schema_version` exists to flag.

## Consequences
- `to_json()` unconditionally emits `tile_size` for every kernel record now
  (`0` for `v0_reference`/`v1_naive`, `32` for `v2_tiled`) -- one schema
  shape for every variant, simpler for any later analysis script to parse.
- `scripts/validate_results.py` requires no changes: it validates against
  whatever the schema currently says, and its one cross-field check
  (`len(raw_timings_ms) == measured_reps`) is unaffected.
- Any future variant that needs its own one-off metadata field (V3's
  running-max/running-sum accumulator width is a plausible future
  candidate) should follow this same pattern -- optional field, `0`/`""`
  "not applicable" default, documented in an ADR here -- rather than
  overloading an existing field's meaning per variant.
