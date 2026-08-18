# ADR 0010: BenchResult Schema Extension for Phase 5 (`peak_memory_bytes` field)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Phase 5's exit criterion is literally "Peak-memory
  scales as designed" (roadmap). Spec SS1.6 also lists a
  "peak-memory-vs-sequence-length plot" among this project's website
  deliverables. Neither is possible to support from committed raw data
  without a benchmark record that actually holds a measured peak-memory
  number -- `BenchResult` (Phase 0) never needed one before, since no
  earlier variant's memory-scaling *behavior* (as opposed to its
  bandwidth/latency) was itself the thing being demonstrated.

## Context
This mirrors ADR 0008 ("BenchResult Schema Extension for Phase 3
(`tile_size` field)") almost exactly: a new variant's benchmark needs one
piece of per-result metadata no earlier variant required, and the schema
needs to grow additively without invalidating any already-committed record.
KernelForge's own ADR 0011 established this same additive-field pattern
first; ADR 0008 already reused it once in this repo, and this ADR reuses
it again.

## Decision
1. Extend `flashlite.bench_schema.BenchResult` with one new field:
   `peak_memory_bytes: float = 0.0`. `0.0` means "not measured for this
   record" (every Phase 0-3 record, and any future record from a sweep that
   only cares about latency) -- same "not applicable"/"not measured"
   sentinel convention ADR 0008 already established for `tile_size`.
2. `scripts/run_benchmarks.py` measures `peak_memory_bytes` via
   `torch.cuda.reset_peak_memory_stats()` immediately before, and
   `torch.cuda.max_memory_allocated()` immediately after (both wrapped in
   `torch.cuda.synchronize()`), ONE additional untimed forward call per
   point in any sweep config that requests it (`benchmarks/configs/`'s
   `"measure_peak_memory": true` flag, opt-in per config rather than
   default-on) -- kept as a clearly separate step from the
   `raw_timings_ms` measurement loop, not interleaved with it, so repeated
   timed calls (which may reuse already-allocated caching-allocator memory
   from earlier reps) cannot be mistaken for a fresh single-call peak.
3. Add `peak_memory_bytes` to `benchmarks/schema/bench_result.schema.json`
   as an **optional** property (`"type": "number", "minimum": 0`), not
   added to `required` -- every existing record remains schema-valid
   unchanged, exactly as ADR 0008 point 3 does for `tile_size`.
4. `schema_version` stays `"1.0"` -- additive, backward-compatible change,
   the identical reasoning ADR 0008 point 4 already gives.

## Consequences
- `to_json()` unconditionally emits `peak_memory_bytes` for every record
  now (`0.0` for records that did not measure it, a real byte count for
  records that did) -- one schema shape for every variant/sweep, simpler
  for any later analysis script to parse, matching ADR 0008's point 1
  consequence for `tile_size`.
- This is what makes
  `tests/correctness/test_fused_attention.py::test_fused_peak_memory_grows_far_slower_than_materializing_variants`'s
  measured claim (checked live, from actual `torch.cuda.max_memory_allocated()`
  calls) reproducible from a *committed benchmark artifact* too, not only
  from a test assertion that runs and is discarded: the Phase 4/5 sweep
  (`benchmarks/configs/phase4_5_comparison.json`) populates this field for
  `v1_naive`/`v2_tiled`/`v3_online_softmax` (all materializing) and
  `v4_fused` (not materializing) across a `seq_len` sweep, so the
  peak-memory-vs-sequence-length comparison spec SS1.6 asks for as a
  website deliverable is literally computable from
  `benchmarks/raw/attention.jsonl`, per this project's hard rule that every
  numeric claim trace back to a raw artifact file.
- Any future variant needing its own one-off metadata field should follow
  this same pattern -- optional field, sentinel default, documented here --
  rather than overloading an existing field's meaning, per ADR 0008's own
  final consequence.
