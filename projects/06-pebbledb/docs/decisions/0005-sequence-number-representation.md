# ADR 0005: Sequence number representation (spec decision D2)

- **Status:** Accepted
- **Phase:** 1 (WAL)
- **Spec decision:** D2 — "Sequence-number representation" — default per
  spec §1.10: **monotonically increasing `u64`**.

## Context

Every WAL record (spec FR10) and, later, every MemTable/SSTable entry needs
a total order that is independent of wall-clock time and independent of key
comparator order, so that "which write happened last for this key" is always
decidable — this is what makes tombstone semantics (spec invariant 4) and
compaction's "preserve the latest visible value" rule (spec invariant 6)
well-defined even when the same key is written multiple times across
different MemTables and SSTables.

## Decision

- **A single global, monotonically increasing `uint64_t` sequence number**,
  assigned once per logical write operation (one `Put` or one `Delete`) and
  stored in the WAL record's `sequence` field (`wal::Record::sequence`,
  `include/pebbledb/wal/record.hpp`).
- **64 bits, not 32.** At any sustained write rate this project's benchmarks
  (spec §1.9) would exercise, 32 bits wraps in a timeframe worth worrying
  about; 64 bits does not in any practical run of this project, and the
  8-byte fixed-width encoding (`util::PutFixed64`/`DecodeFixed64`) costs
  nothing extra to reason about versus a 32-bit field would.
- **Assigned by the writer, not derived from wall-clock time.** Wall-clock
  timestamps are not monotonic across NTP adjustments or clock changes and
  are not a safe ordering key for correctness; a purely logical counter is.
  (Nothing in the current WAL/DB code stamps records with a timestamp at
  all — if one is added later, e.g. for TTL or observability, it is
  additional metadata, never the ordering key.)
- **Encoded fixed-width, little-endian**, like every other WAL record field
  — see `docs/file-format.md` and ADR 0007 for the byte-level layout. This
  keeps sequence-number comparison on decoded records a plain integer
  comparison; there is no varint or other variable-width encoding to strip
  first.
- **Scope for Phase 1:** the WAL format carries and round-trips a caller-
  supplied `sequence` value; `wal::Writer`/`wal::Reader` do not themselves
  allocate or validate sequence numbers (no "must be strictly increasing"
  check exists yet in the WAL layer). `tests/unit/test_wal_record_codec.cpp`
  covers the encoding at the boundary (`SequenceNumberRoundTripsAtUint64Max`).
  A single global counter that actually assigns these values lives with
  `DB` once the WAL is wired into the write path (Phase 2 onward, per the
  roadmap) — this ADR fixes the *representation*, not yet the allocator.

## Consequences

- Any two records for the same key can always be totally ordered by
  comparing `sequence`, independent of which MemTable/SSTable/WAL segment
  they came from — this is the mechanism invariants 4 and 6 (§1.5) are
  built on once MemTable (Phase 2) and compaction (Phase 6) exist.
- The 8-byte field is a fixed, non-negotiable cost per WAL record
  (`kHeaderSize` = 26 bytes total, of which 8 are the sequence number) —
  already accounted for in the file-format documentation and not expected
  to shrink; a future compressed/delta-encoded sequence representation
  would be a new format version (ADR + `docs/file-format.md` update), not a
  silent reinterpretation of v1's field.
