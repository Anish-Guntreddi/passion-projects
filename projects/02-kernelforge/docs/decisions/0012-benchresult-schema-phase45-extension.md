# ADR 0012: BenchResult Schema Extension for Phase 4/5 (GEMM, Softmax, RMSNorm)

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
ADR 0011 extended `kernelforge::BenchResult` / `benchmarks/schema/
bench_result.schema.json` for Phase 2/3's three new families (`reduction`,
`scan`, `histogram`). Phase 4 adds `gemm`; Phase 5 adds `softmax` and
`rmsnorm`. GEMM needs one new piece of per-result metadata no earlier
family required: the shared inner dimension `K` (a GEMM problem is M x N x
K; M and N alone do not fully describe it, and `K` uniquely determines how
many terms each output element's dot product sums — the ladder's whole
"memory reuse" story in `src/kernels/gemm/README.md` is stated in terms of
K-chunks). Softmax and RMSNorm need no new fields: both are row-wise
reductions over an (rows x cols) matrix, and `rows`/`cols` — already
present in the schema for `transpose` — capture that shape with no
GEMM-specific ambiguity (unlike `k`, `rows`/`cols` mean the same thing —
"how many of this axis" — for every family that populates them).

## Decision
1. Extend `BenchResult` with one new field: `int k` (GEMM's inner
   dimension). Defaults to `0` ("not applicable") and is populated only by
   `bench_gemm_main.cpp` — mirroring how `rows`/`cols` (transpose/gemm/
   softmax/rmsnorm) and `num_bins`/`contention_profile` (histogram-only)
   already work per ADR 0010/0011. GEMM reuses `rows`/`cols` for M/N
   (documented in `bench_result.hpp`'s field comments) rather than adding
   `m`/`n` fields that would duplicate what `rows`/`cols` already mean.
2. Extend `kernel_family`'s enum (both the C++ comment and the JSON
   Schema) with `"gemm"`, `"softmax"`, `"rmsnorm"`.
3. Add `k` to `benchmarks/schema/bench_result.schema.json` as an
   **optional** property (not added to `required`) — every existing
   Phase 0-3 record remains schema-valid unchanged.
4. `schema_version` stays `"1.0"`, same additive/backward-compatible test
   ADR 0010/0011 already apply: one new optional field and three new enum
   values, nothing removed/renamed/reinterpreted.
5. GEMM's FLOP-counting convention (recorded here since it is a genuine,
   if small, methodology decision, not just a schema field): `gflops =
   (2 * M * N * K) / (median_ms * 1e6)` — one multiply and one add per
   inner-product term, the standard convention used by cuBLAS's own
   documentation and every GEMM benchmarking write-up this repo's authors
   are aware of, so this repo's numbers are comparable to numbers reported
   elsewhere using the same convention.

## Consequences
- `to_json()` unconditionally emits `k` for every kernel family now (same
  pattern as `rows`/`cols`/`num_bins`/`contention_profile`), so every
  committed record keeps the same top-level shape.
- `scripts/validate_results.py` requires no changes, same reasoning as
  ADR 0011.
- GEMM's `bytes_moved` (and therefore `effective_bandwidth_gb_s`) is
  populated with the MINIMUM unique working-set size (`(M*K + K*N + M*N)
  * sizeof(float)`), not actual DRAM traffic — GEMM is compute-bound, not
  bandwidth-bound, for every rung and size this repo benchmarks, so
  `gflops` (not `effective_bandwidth_gb_s`) is this family's primary
  metric (FR4: "effective bandwidth OR FLOP/s where meaningful"); the
  bandwidth field is populated anyway for schema uniformity but is not
  the number `src/kernels/gemm/README.md`'s narrative is built on.
- Any future kernel family that needs its own one-off metadata field
  should follow this same pattern (optional field, defaults to a clear
  "not applicable" sentinel, documented in an ADR here) rather than
  overloading an existing field's meaning per family — same closing note
  ADR 0011 left for the family after it.
