# ADR 0011: BenchResult Schema Extension for Phase 2/3 (Reduction, Scan, Histogram)

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
ADR 0010 defined `kernelforge::BenchResult` / `benchmarks/schema/bench_result.schema.json`
for Phase 0/1's four kernel families (`vector_add`, `saxpy`, `transpose`,
`stride_copy`). Phase 2 adds `reduction` and `scan`; Phase 3 adds
`histogram`. Histogram benchmarks need two new pieces of per-result
metadata that no earlier family required: `num_bins` (the histogram width)
and which input-distribution ("contention profile") produced the result,
since Phase 3's entire deliverable is a **documented comparison** between
low- and high-contention runs of the same kernel/size (see
`src/kernels/histogram/README.md`) — that comparison is meaningless if a
committed result doesn't say which distribution generated it.

## Decision
1. Extend `BenchResult` with two new fields: `int num_bins` and
   `std::string contention_profile`. Both default to "not applicable"
   values (`0` and `""` respectively) and are populated only by the
   histogram benchmark binary — mirroring how `rows`/`cols` (transpose-only)
   and `stride` (stride_copy-only) already work in the Phase 0/1 schema.
2. Extend `kernel_family`'s enum (both the C++ comment and the JSON Schema)
   with `"reduction"`, `"scan"`, `"histogram"`.
3. Add both new fields to `benchmarks/schema/bench_result.schema.json` as
   **optional** properties (not added to the `required` array) — every
   existing Phase 0/1 record remains schema-valid unchanged.
4. `schema_version` stays `"1.0"`. Per ADR 0010, that field exists to make
   *incompatible* changes to the schema detectable by readers; this change
   adds two new optional fields and three new enum values without removing,
   renaming, or changing the meaning of anything a Phase 0/1 reader already
   depends on, so it is additive/backward-compatible by the same test ADR
   0010 implicitly applies, and does not warrant a version bump.

## Consequences
- `to_json()` unconditionally emits `num_bins`/`contention_profile` for
  every kernel family now (same pattern as `rows`/`cols`/`stride`), so
  every committed record has the same top-level shape regardless of family
  — simpler for the Phase 7 analysis pipeline (FR7) to parse with one
  schema instead of a family-specific one.
- `scripts/validate_results.py` requires no changes: it validates against
  whatever `benchmarks/schema/bench_result.schema.json` currently says,
  and the cross-field `raw_timings_ms` length check it also performs is
  unaffected by this extension.
- Any future kernel family that needs its own one-off metadata field
  should follow this same pattern (optional field, defaults to a clear
  "not applicable" sentinel, documented in an ADR here) rather than
  overloading an existing field's meaning per family.
