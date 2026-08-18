# ADR 0009: SSTable block size and compression (spec decision D5)

- **Status:** Accepted (filter-block reservation superseded by ADR 0014
  at Phase 5 — see note at the end of this document; block size and
  compression decisions below are unaffected and remain current)
- **Phase:** 3 (SSTable)
- **Spec decision:** D5 — "Table block size / compression" — default per
  spec §1.10: **4 KB blocks; compression is stretch.**

## Context

An SSTable (spec FR3) needs to be split into blocks rather than treated as
one giant sorted run, so a point lookup can `pread` just the one block
that might contain a key instead of reading (and holding in memory) the
whole file — this is exactly what the "sparse/block index" requirement in
FR3 is for. Block size is a trade-off: smaller blocks mean a finer-grained
index (more entries to search/hold, more per-block fixed overhead — the 5-
byte trailer, ADR 0010) but less wasted I/O per lookup; larger blocks mean
a smaller index but more bytes read (and decoded) per lookup than
strictly necessary. Compression (Snappy/Zstd) is explicitly listed as a
stretch goal in spec Part 3, not an MVP requirement.

## Decision — 4 KiB target block size, not a hard cap

`sstable::Writer::Options::target_block_size_bytes` defaults to 4096
(`include/pebbledb/sstable/writer.hpp`), the spec's default. It is a
*target*, checked after each `Add()`: once the current block's
accumulated encoded-entry bytes reach this size, the block is closed and
a new one started. It is deliberately not a hard cap — a single entry
whose key+value exceeds 4 KiB on its own is still written whole into its
own block rather than being split across two blocks, since splitting a
single logical entry across two blocks would require additional
continuation-record machinery (LevelDB's actual approach for very large
values) that is not worth the complexity at this project's scope; see
`tests/integration/test_sstable_writer_reader.cpp`'s
`LargeKeyAndValueRoundTrip` for confirmation this works correctly with
oversized single entries. 4 KiB matches common OS page/filesystem-block
sizes, which is the conventional reasoning behind this being the
ecosystem-standard default (LevelDB/RocksDB use the same figure) —
PebbleDB does not measure this trade-off itself before Phase 9's
benchmark study exists to do so; 4 KiB is adopted as the spec's stated
default, not as a value this project has independently tuned.

`target_block_size_bytes` is a per-`Writer::Options` field, not a
compile-time constant, so a future benchmark (Phase 9) can vary it without
a rebuild, and so `tests/integration/test_sstable_writer_reader.cpp`'s
`KeysSpanningManyBlocksAreAllFoundCorrectly` and
`tests/corruption/test_sstable_corruption.cpp` can force many small blocks
(64/48-byte targets) to exercise the multi-block index/binary-search path
and per-block corruption isolation without needing a large, slow dataset
to reach that code path naturally.

## Decision — no compression in this version; the format reserves a byte for it

`sstable::CompressionType` (`include/pebbledb/sstable/format.hpp`) has
exactly one value, `kNone = 0`, which is the only value this version's
`Writer` ever emits and the only value this version's `Reader` accepts —
anything else decodes as `DecodeStatus::kUnsupportedCompression`
(`tests/unit/test_sstable_format.cpp`'s
`VerifyRejectsUnsupportedCompressionByte`), reported through
`sstable::Reader` as `Status::NotSupported`, distinct from
`Status::Corruption` (mirrors `wal::ReadResult::kUnsupportedVersion`'s
rationale, ADR 0006: a structurally-recognized-but-unhandled value is not
evidence of bit rot).

The block trailer format (ADR 0010) still spends one byte on
`compression_type` even though it is always `kNone` today. This is a
deliberate, low-cost forward-compatibility reservation: it means a future
Snappy/Zstd implementation (stretch goal) is a *content* change (new
`CompressionType` value, decompress-before-decode in the reader) rather
than a *layout* change (no new field, no `format_version` bump, no
`docs/file-format.md` structural rewrite) — the same reasoning behind
reserving the filter block region below.

## Decision — filter block: reserved, always empty in this version

The footer (ADR 0010) unconditionally carries a `filter_handle:
BlockHandle`, which this version's `Writer::Finish()` always sets to
`{offset, 0}` (zero size). A `Reader` must treat `size == 0` as "no filter
block present" and never attempt to read it — this is exactly how Phase 5's
Bloom filter (spec roadmap Phase 5, FR3's "per-table (or block-group)
Bloom filter") will slot in later: `Writer` starts actually building and
writing a non-empty filter block, `Reader` starts consulting it before a
data-block read to skip provably-absent keys, and neither side needs a new
`format_version` or a `docs/file-format.md` layout rewrite to do it — only
new field *content*, on a field the footer has carried since Phase 3.
This mirrors ADR 0004's own note that later phases are expected to extend
this format rather than replace it. Choosing to reserve this now (rather
than add the field only when Phase 5 begins) was a deliberate,
low-complexity call: it costs 16 bytes in every footer and one
`{0, 0}` assignment in `Writer::Finish()`, in exchange for one fewer future
format version to design, document, and test migration for.

## Consequences

- Benchmark numbers (Phase 9) that vary block size must report which
  `target_block_size_bytes` was used, the same discipline ADR 0006
  established for WAL sync policy — block size measurably trades index
  overhead against per-lookup I/O, and comparing results across different
  block sizes without labeling would be as misleading as comparing across
  sync policies.
- Every SSTable this version's `Writer` produces has `filter_block.size ==
  0`; `tools/inspect_sstable` prints `(absent -- reserved for Phase 5's
  Bloom filter)` next to that field specifically so this is visible, not
  silently implied, when inspecting a real file. **(Historical — see the
  note below.)**
- Snappy/Zstd block compression, prefix/key compression, and a skip-list
  MemTable (ADR 0008) remain the documented stretch goals they were before
  this ADR (spec Part 3) — nothing here implements them, only reserves
  the on-disk room for compression to be added later without a format
  break.

## Update (Phase 5) — filter block reservation superseded by ADR 0014

As anticipated by this ADR's own "Decision — filter block: reserved,
always empty in this version" section, Phase 5 (`docs/decisions/0014-bloom-filter-design.md`)
fills that region with real Bloom filter content — exactly the
content-only, no-layout-change, no-`format_version`-bump extension this
ADR predicted. `sstable::Writer::Options::filter_bits_per_key` now
defaults to 10 (a real filter is built and written by default);
`filter_bits_per_key == 0` reproduces this ADR's original "always empty"
behavior exactly. `tools/inspect_sstable`'s "(absent -- reserved for
Phase 5's Bloom filter)" message referenced above is itself superseded —
see ADR 0014's Consequences section for the current output. This note is
appended rather than rewriting the sections above, which remain an
accurate historical record of the Phase 3 decision as originally made.
