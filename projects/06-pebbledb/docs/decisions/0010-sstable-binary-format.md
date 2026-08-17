# ADR 0010: SSTable v1 binary format — layout and encoding choices

- **Status:** Accepted
- **Phase:** 3 (SSTable)
- **Spec requirement:** FR3 — "Sorted table (SSTable) file format with
  documented binary layout (conceptually: data blocks | index block |
  bloom/filter block | footer/metadata); sparse/block index; per-table (or
  block-group) Bloom filter; file inspector tools." FR10 — "All binary
  formats versioned from the beginning."

This ADR covers the SSTable v1 format's *layout and encoding* choices —
the SSTable counterpart of ADR 0007's role for the WAL format. The
field-by-field on-disk layout table is the single source of truth in
`docs/file-format.md`; this ADR explains *why* those choices were made,
which `docs/file-format.md` deliberately does not re-litigate. See ADR
0009 for block size and the reserved filter-block/compression-byte
decisions specifically, and ADR 0008 for the MemTable this format is
designed to receive a flush from.

## Decision — reuse the WAL's byte-level conventions

Every multi-byte field is fixed-width, explicitly little-endian
(`util::PutFixed32`/`PutFixed64`), and every self-contained unit (data
block, index block, footer) carries its own CRC-32C checksum — exactly
ADR 0007's conventions for the WAL, reused rather than re-derived. This
was ADR 0007's own stated expectation ("SSTable blocks (Phase 3)... are
expected to reuse the same fixed-width/little-endian/CRC-32C conventions
established here for consistency"), and reusing a proven, already-tested
encoding toolkit (`util::coding`, `util::crc32c`) rather than inventing a
second one is the obvious choice at this project's scope.

## Decision — file layout: data blocks, then filter block, then index block, then footer

```
[data block 0] [data block 1] ... [data block N-1]
[filter block]   (reserved, zero-length in this version -- ADR 0009)
[index block]
[footer]          (fixed kFooterSize = 49 bytes, always the file's last
                   49 bytes)
```

The footer is fixed-size and always at the very end of the file, so a
reader locates everything else with exactly one extra seek+pread: read the
last 49 bytes first, then use the offsets/sizes it contains to find the
index block (and, from there, individual data blocks). This is the
standard SSTable/footer pattern (LevelDB, RocksDB, and this project's own
WAL-adjacent design all favor "self-describing trailer, not a fixed
leading header") and is what makes `sstable::Reader::Open` an O(1)-seek
operation regardless of file size, rather than needing to scan from the
front.

Data blocks come first, in write order, immediately followed by the
(always-empty, in this version) filter block region and then the index
block — see ADR 0009 for why the filter block is reserved-but-empty here.

## Decision — data block entries: fixed 17-byte header, no restart points

Each entry within a data block is `entry_type(1) + sequence(8) +
key_length(4) + value_length(4)` (17 bytes, `kEntryHeaderSize`) followed
by the raw key and value bytes — deliberately the same shape as a WAL
record's header (ADR 0007), minus the fields a block-relative entry does
not need (no `magic`/`format_version` per entry — those live once, in the
block/footer trailers that wrap many entries at a time; no per-entry
checksum — one checksum covers the whole block, see below).

**No restart points / prefix compression.** LevelDB's block format shares
common key prefixes between consecutive entries (a "restart point" scheme)
to save space, at the cost of a decode step that must track the previous
key. Spec Part 3 lists "prefix/key compression" as an explicit stretch
goal, not an MVP requirement — PebbleDB's Phase 3 blocks store every
entry's key and value in full, decoded by a straight-line, stateless
`while (!view.empty()) { DecodeEntry(&view, &entry); ... }` loop
(`sstable::DecodeEntry`, reused identically by `Reader::Get`'s
in-block scan and `Reader::ForEachEntry`'s full traversal). This trades
some on-disk size for meaningfully simpler, more directly testable decode
logic — the right trade at this project's scope, and reversible later
(a new block encoding, gated by a `format_version` bump) without touching
anything above the block layer.

**No separate sanity-bound check on `key_length`/`value_length`,
unlike `wal::DecodeHeader`'s `kMaxKeyLength`/`kMaxValueLength`.** The WAL
reader decodes a fixed header off disk and then must decide, from a
possibly-corrupted length field, how many *additional* bytes to read from
disk — an unbounded length there risks an unbounded read attempt, which is
exactly what those constants guard against (ADR 0006,
`tests/corruption/test_wal_corruption.cpp`'s
`Invariant7_OversizedDeclaredLengthIsRejectedNotReadAsHugeAllocation`).
An SSTable block, by contrast, is always read as one bounded, already-
in-memory buffer (sized by its `BlockHandle::size`, itself already
checksum-verified before any entry is decoded) *before* `DecodeEntry` is
ever called — a corrupted length field inside that buffer can only ever
cause `input->size() < total`, which `DecodeEntry` already checks and
reports as `DecodeStatus::kTruncated`. There is no unbounded-disk-read or
unbounded-allocation path to guard against separately; see
`tests/corruption/test_sstable_corruption.cpp`'s
`Invariant7_OversizedIndexKeyLengthIsRejectedNotReadAsHugeAllocation` for
the analogous case on the index side, confirming this reasoning holds in
practice, not just in the abstract.

## Decision — block trailer: one checksum per block, not per entry

Every block (data or index — both use the exact same trailer,
`sstable::FinishBlock`/`VerifyAndStripBlockTrailer`) is followed by a
5-byte trailer: `compression_type(1)` (ADR 0009) + `checksum(4)`, where
the checksum covers the block's raw entry bytes *and* the
`compression_type` byte (everything in the block except the checksum
field itself — the same "checksum covers everything after itself" shape
as the WAL record format and the footer, below). One checksum per block
rather than one per entry is a deliberate space/verification-granularity
trade: it means a single corrupted byte anywhere in a block fails *every*
lookup that would read that block (not just the one entry whose bytes were
actually touched), but at 4 KiB blocks (ADR 0009) holding many entries
each, the overhead of a checksum per entry (4 extra bytes × every entry,
versus 4 bytes × every ~4 KiB) would be disproportionate at this format's
scope. This is the same reasoning LevelDB/RocksDB apply to their own
block trailers.

**Corruption blast radius, by design:** because `sstable::Reader` has no
single sequential cursor the way `wal::Reader` does — every `Get()`
independently `pread`s and verifies only the block(s) it needs — a
corrupted data block only fails lookups that land in *that* block; lookups
resolved by other, uncorrupted blocks in the same file are unaffected
(`tests/corruption/test_sstable_corruption.cpp`'s
`OtherBlocksRemainReadableAfterOneBlockIsCorrupted`). This is a materially
different failure shape from the WAL's "one corruption point ends all
further replay" policy (ADR 0006) — and a deliberately correct one for an
SSTable specifically, since (unlike the WAL, which is replayed strictly in
write order to reconstruct one linear history) a table's blocks are
independent, randomly-addressable units with no ordering dependency
between them.

## Decision — index block: last-key separator, sparse, loaded eagerly

Each index entry pairs a data block's *last* (largest) key with that
block's `BlockHandle` (`key_length(4) + key bytes + offset(8) + size(8)`).
Because data blocks are written in strictly increasing key order and never
overlap, "last key in block N" is a valid separator: the first index entry
whose key is `>=` a target key identifies the one data block that could
contain it (`std::lower_bound` over the in-memory index,
`sstable::Reader::Get`). This is the standard SSTable sparse-index
approach (one entry per block, not per key — "sparse" per FR3), kept
small enough (by construction: at most one index entry per ~4 KiB data
block) that `Reader::Open` loads the *entire* index block into memory
unconditionally, once, rather than treating it as another randomly-paged
structure. This is also why index-block corruption is caught eagerly, at
`Open()` time, rather than resurfacing unpredictably on whichever later
`Get()` call happens to need it — a deliberate choice matching the WAL
reader's own preference for surfacing structural problems as early and
predictably as possible.

## Decision — footer: magic, version, two block handles, entry count

`checksum(4) + magic(4) + format_version(1) + index_handle(16) +
filter_handle(16) + num_entries(8)` = 49 bytes (`kFooterSize`). `magic =
0x50425354` — read as big-endian bytes `50 42 53 54`, spelling ASCII
`"PBST"` ("PebbleDB Sorted Table"), the same hex-dump-mnemonic convention
ADR 0007 chose for the WAL's `"PBK1"`. `format_version` and the checksum
follow the WAL footer/header's own validation order rationale exactly:
magic first, then version (an unrecognized-but-structurally-valid future
version is reported as `DecodeStatus::kUnsupportedVersion` →
`Status::NotSupported`, never `Status::Corruption` — mirrors ADR 0006's
rationale for `wal::ReadResult::kUnsupportedVersion`), then the checksum
last (needs the whole body already read, unlike the WAL's header/payload
split, since the footer is always read as one complete, fixed-size
region). `num_entries` is carried for `tools/inspect_sstable` and future
compaction/stats code to consult without needing a full table scan.

## Consequences

- `sstable::Reader::Open` is exactly two structural reads before it can
  serve a `Get()`: the footer (fixed offset from EOF), then the index
  block (offset/size from the footer) — both eagerly verified. Every
  `Get()` after that is exactly one more read (the one candidate data
  block), located by an in-memory binary search over the already-loaded
  index.
- Any change to block/footer/index layout, checksum coverage, or the
  entry header shape is a new `format_version` and a `docs/file-format.md`
  update — never a silent reinterpretation of v1's bytes, exactly as ADR
  0007 established for the WAL and as `DecodeFooter`/`DecodeEntry`'s
  version/bounds checks already enforce mechanically, not just by
  documentation convention.
- Phase 4's manifest format and Phase 6's compaction are expected to keep
  reusing these same conventions (fixed-width little-endian fields,
  CRC-32C, a self-describing trailer/footer) for the same reasons ADR
  0004 anticipated — documented in their own ADRs and `docs/file-format.md`
  sections when those phases begin.
