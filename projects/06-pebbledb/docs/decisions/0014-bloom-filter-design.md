# ADR 0014: Bloom filter design — content format, hashing, sizing

- **Status:** Accepted
- **Phase:** 5 (Bloom filter + block cache)
- **Spec requirement:** FR3 — "...per-table (or block-group) Bloom
  filter..." Roadmap Phase 5 deliverable: "Negative-read filtering."
  Exit criterion: "Benchmark demonstrates reduced unnecessary reads /
  cache effects."
- **Related:** ADR 0009 (reserved the SSTable footer's `filter_handle`
  field and the filter-block region back in Phase 3, specifically so
  this phase could fill it in without a format-version bump).

## Context

`sstable::Writer`/`Reader` (Phase 3) already carry a `filter_handle:
BlockHandle` in every footer, unconditionally set to `{offset, 0}` (ADR
0009's deliberate reservation). This phase had to decide: per-table or
per-block-group filtering (FR3 explicitly allows either); what goes
*inside* the filter block once it is non-empty; which hash construction;
and how many bits per key / probes to use.

## Decision — one filter per table, not per block-group

`bloom::BuildFilter()` (`include/pebbledb/bloom/bloom_filter.hpp`,
`src/bloom/bloom_filter.cpp`) takes the full list of keys `Writer` has
seen (`keys_for_filter_`, accumulated across every `Add()` call) and
builds one filter covering the whole table. A per-block-group filter
(a separate small filter per data block, LevelDB/RocksDB's actual
design) would give finer-grained negative answers at a smaller
per-lookup cost, but requires the filter-block builder to track block
boundaries and store an offset table mapping a data block to its
filter's byte range — meaningfully more machinery for a proportionally
small benefit at this project's block sizes (ADR 0009: ~4 KiB target,
so a table of even a few hundred KiB already has very few blocks). A
single per-table filter is simpler to build (buffer keys, one
`BuildFilter()` call in `Finish()`), simpler to store (one `BlockHandle`,
already present in the footer), and simpler to reason about — the right
trade at this project's scope, matching the "one simple, correct choice
over a complex matrix" preference spec Part 3's non-goals establish
elsewhere.

## Decision — filter-block content format: `num_probes(1) + bits`

The filter block's *content* (before `sstable::FinishBlock()` wraps it
with the standard compression-type + CRC-32C trailer every block already
gets — ADR 0010) is:

| Field | Size | Description |
|---|---|---|
| `num_probes` | `uint8` | Number of hash probes (k) used to build this filter. |
| `bits` | remaining bytes | The bit array itself. |

Storing `num_probes` explicitly (rather than assuming a fixed,
hardcoded k) makes each filter self-describing — a reader never needs to
assume a particular `bits_per_key`/k configuration was used to build the
specific file it has open, matching this project's "structural fields
before content" convention elsewhere (the WAL's `format_version`, the
manifest's `num_tables`). This also means `Options::filter_bits_per_key`
can vary per-table (e.g. a future benchmark comparing bits-per-key
settings) without any format change.

Wrapping this content with the *existing* block trailer format (rather
than inventing a separate one) means the filter block gets a checksum
for free, the same as every data/index block — a corrupted filter block
is detected at `Reader::Open()` time (loaded eagerly, like the index
block) rather than silently producing wrong answers.

## Decision — hashing: one CRC-32C evaluation, then Kirsch/Mitzenmacher double hashing

Rather than implement a second, bespoke hash function purely for the
Bloom filter, `bloom::Hash1()` reuses `util::Crc32c()` — already
implemented, tested, and used throughout this codebase for on-disk
checksums. A second probe position is derived cheaply from the first via
a bit rotation (`Hash2(h1) = (h1 >> 15) | (h1 << 17)`), and probe `i`'s
bit position is `g_i(x) = h1(x) + i * h2(x) mod num_bits`. This is the
standard **Kirsch/Mitzenmacher "double hashing"** construction (Kirsch &
Mitzenmacher, "Less Hashing, Same Performance: Building a Better Bloom
Filter," 2006): it gives `k` practically-independent probe positions
from a single real hash evaluation per key, with (per that paper) the
same asymptotic false-positive rate as `k` fully independent hash
functions — avoiding both a second from-scratch hash implementation and
the cost of `k` separate hash evaluations per key. `k` (`ChooseNumProbes`)
is the standard `bits_per_key * ln(2)` optimal-probes formula, rounded
and clamped to `[1, 30]` — the same clamp range used by essentially
every mainstream Bloom filter implementation, guarding against a
degenerate `k = 0` (no filtering at all) or an impractically large probe
count for a very large `bits_per_key`.

CRC-32C is not a cryptographic hash and was never chosen for that
property — its avalanche behavior (small input changes flip roughly half
the output bits) is good enough for bit-selection purposes, which is all
a Bloom filter probe position needs.

## Decision — default 10 bits per key (~1% false-positive rate)

`FilterPolicy::bits_per_key` defaults to 10, matching the widely used
LevelDB/RocksDB default. At the optimal `k` for 10 bits/key (`k ≈ 6.93`,
clamped to 7 by `ChooseNumProbes`), the standard Bloom filter
false-positive-rate formula `(1 - e^(-kn/m))^k` (with `m/n = 10`)
predicts roughly 1%. `bits_per_key == 0` disables filter generation
entirely for a given `sstable::Writer::Options` — `BuildFilter()` then
returns an empty string, and `Writer::Finish()` falls back to exactly
the same zero-length, reserved filter block every pre-Phase-5 table
already produced (ADR 0009); a reader treats `filter_handle.size == 0`
identically either way ("no filter present," not an error).

## Decision — a malformed or absent filter degrades to "do the real lookup," never to a false negative

`bloom::KeyMayMatch()` returns `true` ("may be present") for an empty or
too-short-to-parse filter, rather than ever risking a false negative
from corrupted/missing filter bytes. This is a correctness-preserving
default: the entire point of a Bloom filter here is to *skip*
unnecessary data-block reads, never to replace them as the source of
truth — a filter answering "maybe" when it cannot answer confidently
simply means this specific optimization does not fire for this lookup,
falling through to the same index-search-then-block-read path a
pre-Phase-5 (or filter-disabled) table always uses. `sstable::Reader::Get()`
only ever uses a *negative* answer to skip a block read it would
otherwise have had to do anyway (see `filter_checks()`/
`filter_rejections()` — Phase 5's benchmark evidence,
`tests/integration/test_sstable_filter_and_cache.cpp`); it never uses a
*positive* answer as license to skip anything.

## Consequences

- `tests/unit/test_bloom_filter.cpp` covers the correctness contract
  (`NoFalseNegativesForAddedKeys` — every key actually added always
  reports "may be present," checked against 2000 keys), a false-positive
  rate sanity bound (`FalsePositiveRateIsRoughlyInExpectedRange` — a
  generous 10% ceiling around the ~1% expected rate, robust against any
  single unlucky draw), and the degrade-safely behavior for
  empty/malformed filters.
- `tests/integration/test_sstable_filter_and_cache.cpp`'s
  `FilterAvoidsDataBlockReadsForMostAbsentKeys` (paired with its control,
  `WithoutFilterEveryAbsentLookupWithinRangeReadsADataBlock`) is this
  phase's direct, deterministic evidence for the roadmap exit criterion
  — measured via `sstable::Reader::data_blocks_read()`, not a wall-clock
  timing benchmark (which would be noisy/non-reproducible as a pass/fail
  test).
- `tools/inspect_sstable` reports a present filter block's bit count
  (derived from `filter_handle.size`) rather than the old, now-outdated
  "(absent -- reserved for Phase 5's Bloom filter)" message.
- A future Phase 6 (compaction) that merges multiple tables into one new
  table can build that new table's filter the same way any other
  `Writer` does — `Add()` every surviving key, `Finish()` builds the
  filter over exactly those keys — no format or algorithm change needed
  here.
