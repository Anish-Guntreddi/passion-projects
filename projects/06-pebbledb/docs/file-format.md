# Binary file formats

This is the single source of truth for PebbleDB's on-disk binary layouts.
Code comments (`include/pebbledb/wal/record.hpp`, `src/wal/record.cpp`, ...)
point here rather than duplicating the field table; keep this document and
the code in sync in the same change whenever either changes (spec Part 4
rule 2: "every binary-format task requires format documentation... updated
in the same change").

## WAL record — format version 1

Encoding rationale (byte order, integer width, checksum algorithm/coverage):
ADR 0007. Sequence-number representation: ADR 0005. Sync-policy and
resync-on-corruption semantics around this format: ADR 0006 and
`docs/durability.md`.

Every field is fixed-width and little-endian
(`util::PutFixed32`/`PutFixed64`, `include/pebbledb/util/coding.hpp`). A
record is a fixed-size header immediately followed by the variable-length
key and value payload — no padding or alignment between fields.

### Header layout (26 bytes, `wal::kHeaderSize`)

| Offset | Size | Field | Type | Description |
|---:|---:|---|---|---|
| 0 | 4 | `checksum` | `uint32` | CRC-32C (Castagnoli) over every byte from offset 4 (`magic`) through the end of `value` — i.e. every field below plus the key/value payload, but **not** this field itself. |
| 4 | 4 | `magic` | `uint32` | Constant `wal::kMagic` = `0x50424B31` — read as big-endian bytes `50 42 4B 31`, these spell ASCII `"PBK1"` ("PebbleDB Kv format 1"), a human-recognizable mnemonic for hex-dump debugging even though the field itself is stored little-endian on disk (bytes `31 4B 42 50`, per the worked example below). Identifies these bytes as a PebbleDB WAL v1 record; a mismatch is `DecodeStatus::kBadMagic`. |
| 8 | 1 | `format_version` | `uint8` | Constant `wal::kFormatVersion` = `1` for this document. `0` or any value greater than the reader's `kFormatVersion` is `DecodeStatus::kUnsupportedVersion` (checked before checksum verification, since a future version's checksum coverage/layout is not necessarily this one). |
| 9 | 1 | `op` | `uint8` | `1` = `RecordType::kPut`, `2` = `RecordType::kDelete`. Any other value is `DecodeStatus::kBadOperation` — but this check happens **after** checksum verification (see "Validation order" below). |
| 10 | 8 | `sequence` | `uint64` | Monotonically increasing sequence number for this write (ADR 0005). Not itself validated for strict monotonicity by the WAL layer (see ADR 0005, "Scope for Phase 1"). |
| 18 | 4 | `key_length` | `uint32` | Length of `key` in bytes. Must be `<= wal::kMaxKeyLength` (1 MiB = `1u << 20`); otherwise `DecodeStatus::kBadLength`. |
| 22 | 4 | `value_length` | `uint32` | Length of `value` in bytes. Must be `<= wal::kMaxValueLength` (64 MiB = `1u << 26`); otherwise `DecodeStatus::kBadLength`. **Always 0 for a `kDelete` record** — `EncodeRecord` ignores any bytes in `Record::value` when `type == kDelete` and encodes `value_length` as `0` regardless of what the caller set. |

Total fixed header size: **26 bytes** (`checksum` 4 + `magic` 4 +
`format_version` 1 + `op` 1 + `sequence` 8 + `key_length` 4 +
`value_length` 4).

### Payload (variable length)

| Field | Size | Description |
|---|---|---|
| `key` | `key_length` bytes | Binary-safe key bytes, immediately after the header. |
| `value` | `value_length` bytes | Binary-safe value bytes, immediately after `key`. Absent (zero bytes) for a `kDelete` record. |

A full record's on-disk size is therefore `kHeaderSize + key_length +
value_length` bytes. Records are concatenated back-to-back with no
separator, delimiter, or padding — the header's own length fields are what
let a reader find the start of the next record.

### Sanity bounds on declared lengths

`wal::kMaxKeyLength` (1 MiB) and `wal::kMaxValueLength` (64 MiB) bound
`key_length`/`value_length` and are checked by `DecodeHeader` **before**
attempting to read that many payload bytes off disk, so a corrupted length
field (e.g. a flipped high bit turning a small length into ~4 billion)
fails fast as a decode error instead of attempting a multi-gigabyte read
(`tests/corruption/test_wal_corruption.cpp`'s
`Invariant7_OversizedDeclaredLengthIsRejectedNotReadAsHugeAllocation`).
These bounds are generous relative to any workload this project's
benchmarks (spec §1.9) exercise, not a claim about a production maximum
key/value size. `wal::Writer::AddRecord()` enforces the same bounds on the
**write** side, before ever appending or acknowledging a record — see
`docs/durability.md`, "Record-size validation happens before
acknowledgment," and ADR 0006.

### Decode/validation order

1. **`DecodeHeader`** — requires exactly `kHeaderSize` bytes (fewer is
   `kTruncatedHeader`). Checks `magic`, then `format_version`, then the
   length bounds, in that order. Does **not** validate the checksum (needs
   the payload, not yet available at this call) or the `op` byte (deferred
   to `DecodeBody`, see below).
2. **`DecodeBody`** — requires the payload to be at least
   `key_length + value_length` bytes (`kTruncatedPayload` otherwise, without
   attempting checksum verification on a known-incomplete payload). Then
   verifies the checksum over `header[4..26) + payload`. Only **after** a
   checksum match does it validate the `op` byte
   (`kOk`/`kPut`/`kDelete` — anything else is `kBadOperation`).
   - **Why op-byte validation is deferred until after the checksum check:**
     a corrupted `op` byte is, by construction, also a corrupted byte
     within the checksum's coverage — so it will already fail the checksum
     check. Validating `op` first would create two possible failure modes
     for the same underlying event (a flipped bit landing on the `op`
     byte) depending on validation order; deferring it makes "any single
     corrupted byte in a fully-present record" report consistently as
     `kChecksumMismatch`, and reserves `kBadOperation` specifically for the
     narrower case of a record whose bytes are *internally self-consistent*
     (checksum matches) but whose `op` value was never valid to begin with
     — see `tests/unit/test_wal_record_codec.cpp`'s
     `DecodeBodyRejectsInvalidOperationByte`, which hand-crafts exactly
     that case.

### Worked example

`EncodeRecord` for a `Put("k", "v")` with `sequence = 1`:

```
offset  bytes                          field
0       <crc32c of bytes 4..28>       checksum
4       31 4b 42 50                   magic (0x50424B31, little-endian)
8       01                            format_version = 1
9       01                            op = kPut
10      01 00 00 00 00 00 00 00       sequence = 1
18      01 00 00 00                   key_length = 1
22      01 00 00 00                   value_length = 1
26      6b                            key = "k"
27      76                            value = "v"
```

Total size: 28 bytes (26-byte header + 1-byte key + 1-byte value).

## SSTable — format version 1

Encoding, layout, and validation-order rationale: ADR 0010. Block size and
compression/filter-block reservation: ADR 0009 (spec decision D5). The
MemTable this format is designed to receive a flush from: ADR 0008 (spec
decision D4).

Every multi-byte field is fixed-width and little-endian
(`util::PutFixed32`/`PutFixed64`), and every self-contained region (data
block, index block, footer) carries its own CRC-32C checksum — the same
conventions as the WAL format above.

### File layout

```
[data block 0] [data block 1] ... [data block N-1]
[filter block]   (reserved for Phase 5's Bloom filter; zero-length in
                   this version's writer output -- ADR 0009)
[index block]
[footer]          (fixed 49 bytes, always the file's last 49 bytes)
```

A reader locates everything by reading the fixed-size footer first (it is
always the last 49 bytes of the file), then using the offsets/sizes it
contains to find the index block and, from there, individual data blocks
(`sstable::Reader::Open`/`Get`, `src/sstable/reader.cpp`).

### Data block entries (`sstable::EncodeEntry`/`DecodeEntry`)

Each entry within a data block is a fixed 17-byte header immediately
followed by the variable-length key/value payload — no padding, no
separator between consecutive entries.

| Offset (within entry) | Size | Field | Type | Description |
|---:|---:|---|---|---|
| 0 | 1 | `entry_type` | `uint8` | `1` = `EntryType::kValue`, `2` = `EntryType::kTombstone` (`include/pebbledb/entry_type.hpp`). Any other value is `DecodeStatus::kBadEntryType`. |
| 1 | 8 | `sequence` | `uint64` | The write's sequence number (ADR 0005), carried through from the MemTable entry this was flushed from. |
| 9 | 4 | `key_length` | `uint32` | Length of `key` in bytes. |
| 13 | 4 | `value_length` | `uint32` | Length of `value` in bytes. **Always 0 for `kTombstone`** — `EncodeEntry` ignores any bytes given for `value` when `type == kTombstone`, exactly like `wal::EncodeRecord` does for `kDelete`. |
| 17 | `key_length` | `key` | bytes | Binary-safe key bytes. |
| 17 + `key_length` | `value_length` | `value` | bytes | Binary-safe value bytes. Absent for `kTombstone`. |

Total fixed header size: **17 bytes** (`sstable::kEntryHeaderSize`).
Entries within a block are sorted by key, ascending (invariant 3),
enforced by `sstable::Writer::Add()` rejecting any key not strictly
greater than the previously added key.

No separate sanity bound exists on `key_length`/`value_length` (contrast
the WAL's `kMaxKeyLength`/`kMaxValueLength`) — see ADR 0010 for why this
is safe: a block is always decoded from an already-bounded, already-
checksum-verified in-memory buffer, never from an unbounded on-disk
length field driving a fresh disk read.

### Block trailer (`sstable::FinishBlock`/`VerifyAndStripBlockTrailer`)

Every data block *and* the index block (same trailer format for both) is
followed by:

| Field | Size | Description |
|---|---|---|
| `compression_type` | `uint8` | `sstable::CompressionType::kNone` (`0`) — the only value this version emits or accepts; any other value is `DecodeStatus::kUnsupportedCompression`. Reserved for future Snappy/Zstd compression (stretch goal, spec Part 3) — see ADR 0009. |
| `checksum` | `uint32` | CRC-32C over the block's raw entry bytes *and* the `compression_type` byte immediately above (i.e. every byte of the block except this field itself). |

Total trailer size: **5 bytes** (`sstable::kBlockTrailerSize`). A block's
total on-disk size (what a `BlockHandle::size` records, and exactly how
many bytes a reader must `pread` starting at `BlockHandle::offset`) is
therefore `entries_bytes + 5`.

**Corruption blast radius:** because `sstable::Reader` reads each block
independently (no single sequential cursor, unlike `wal::Reader`), a
corrupted block only fails lookups that land in that specific block —
lookups resolved by other, uncorrupted blocks in the same file are
unaffected. See ADR 0010 for the full rationale and
`tests/corruption/test_sstable_corruption.cpp`'s
`OtherBlocksRemainReadableAfterOneBlockIsCorrupted`.

### Index block entries (`sstable::EncodeIndexEntry`/`DecodeIndexEntry`)

The index block is one instance of the same block-with-trailer wrapping
as a data block, but its entries have a different, simpler shape: each
pairs a data block's *last* (largest) key with that block's location.
Because data blocks are non-overlapping and written in increasing key
order, "last key in block N" is a valid separator — the first index entry
whose key is `>=` a target key identifies the one data block that could
contain it (`sstable::Reader::Get`'s `std::lower_bound` over the fully
in-memory-loaded index — FR3's "sparse/block index").

| Offset (within entry) | Size | Field | Type | Description |
|---:|---:|---|---|---|
| 0 | 4 | `key_length` | `uint32` | Length of `last_key` in bytes. |
| 4 | `key_length` | `last_key` | bytes | The largest key present in the referenced data block. |
| 4 + `key_length` | 8 | `block_offset` | `uint64` | Byte offset of the referenced data block within the file. |
| 12 + `key_length` | 8 | `block_size` | `uint64` | Total on-disk size of the referenced data block (entries + its own 5-byte trailer). |

The index block itself is small enough (at most one entry per ~4 KiB data
block, spec decision D5) that `sstable::Reader::Open` loads it fully into
memory, once, unconditionally — index-block corruption is therefore
caught eagerly at `Open()` time rather than resurfacing unpredictably on
some later `Get()` call.

### Footer (49 bytes, `sstable::kFooterSize`)

Always the file's last 49 bytes.

| Offset | Size | Field | Type | Description |
|---:|---:|---|---|---|
| 0 | 4 | `checksum` | `uint32` | CRC-32C over every byte from offset 4 (`magic`) through the end of the footer — every field below, but not this field itself. |
| 4 | 4 | `magic` | `uint32` | Constant `sstable::kMagic` = `0x50425354` — read as big-endian bytes `50 42 53 54`, spelling ASCII `"PBST"` ("PebbleDB Sorted Table"), the same hex-dump-mnemonic convention as the WAL's `"PBK1"`. A mismatch is `DecodeStatus::kBadMagic`. |
| 8 | 1 | `format_version` | `uint8` | Constant `sstable::kFormatVersion` = `1` for this document. `0` or greater than the reader's `kFormatVersion` is `DecodeStatus::kUnsupportedVersion` (checked before the checksum, mirroring the WAL header's validation order). |
| 9 | 8 | `index_block_offset` | `uint64` | Byte offset of the index block. |
| 17 | 8 | `index_block_size` | `uint64` | Total on-disk size of the index block (entries + its 5-byte trailer). |
| 25 | 8 | `filter_block_offset` | `uint64` | Byte offset of the filter block. |
| 33 | 8 | `filter_block_size` | `uint64` | Total on-disk size of the filter block. **Always `0` in this version's writer output** — reserved for Phase 5's Bloom filter (ADR 0009). A reader must treat `size == 0` as "no filter block present", not as an error, and must not attempt to read it. |
| 41 | 8 | `num_entries` | `uint64` | Total number of entries (live values + tombstones) written to this table. |

### Decode/validation order

1. **`DecodeFooter`** — requires exactly `kFooterSize` bytes (fewer is
   `kTruncated`). Checks `magic`, then `format_version`, then the
   checksum (the whole footer is already available as one fixed-size
   buffer, unlike the WAL's header/payload split, so there is no
   "checksum needs data not yet read" ordering constraint here — magic
   and version are still checked first purely to keep the same
   fail-fast-on-structural-fields-first shape as the WAL format).
2. **`VerifyAndStripBlockTrailer`** — requires at least `kBlockTrailerSize`
   bytes (fewer is `kTruncated`). Checks the `compression_type` byte, then
   the checksum.
3. **`DecodeEntry`** / **`DecodeIndexEntry`** — requires at least the
   fixed header size, then requires the declared `key_length`/
   `value_length` to fit within the already-supplied (and, for a real
   on-disk read, already checksum-verified) buffer; a declared length that
   would overrun the buffer is `kTruncated`, not a separate "bad length"
   status, since (unlike the WAL) there is no unbounded-disk-read
   consequence to distinguish from an ordinary truncation — see ADR 0010.

### Worked example

`sstable::Writer::Finish()` for a table containing one entry,
`Put("k", "v")` at `sequence = 1`:

```
Data block 0 (entries: 17-byte header + 1-byte key + 1-byte value = 19 bytes;
              block total: 19 + 5-byte trailer = 24 bytes)
  offset 0   01                             entry_type = kValue
  offset 1   01 00 00 00 00 00 00 00        sequence = 1
  offset 9   01 00 00 00                    key_length = 1
  offset 13  01 00 00 00                    value_length = 1
  offset 17  6b                             key = "k"
  offset 18  76                             value = "v"
  offset 19  00                             compression_type = kNone
  offset 20  <crc32c of bytes 0..20>        checksum

Filter block: zero-length (offset 24, size 0 -- nothing written)

Index block (1 index entry: 4 + 1 + 16 = 21 bytes; block total: 21 + 5 = 26 bytes)
  offset 24  01 00 00 00                    key_length = 1
  offset 28  6b                             last_key = "k"
  offset 29  00 00 00 00 00 00 00 00        block_offset = 0
  offset 37  18 00 00 00 00 00 00 00        block_size = 24
  offset 45  00                             compression_type = kNone
  offset 46  <crc32c of bytes 24..46>       checksum

Footer (49 bytes, offset 50)
  offset 50  <crc32c of bytes 54..99>       checksum
  offset 54  54 53 42 50                    magic (0x50425354, little-endian)
  offset 58  01                             format_version = 1
  offset 59  18 00 00 00 00 00 00 00        index_block_offset = 24
  offset 67  1a 00 00 00 00 00 00 00        index_block_size = 26
  offset 75  18 00 00 00 00 00 00 00        filter_block_offset = 24
  offset 83  00 00 00 00 00 00 00 00        filter_block_size = 0
  offset 91  01 00 00 00 00 00 00 00        num_entries = 1
```

Total file size: 99 bytes (24-byte data block + 26-byte index block +
49-byte footer).

## Manifest format

Not yet implemented (Phase 4 per the roadmap). This section is added,
with the same field-by-field table format as above, in the same change
that introduces the format — per spec Part 4 rule 2, a binary-format task
is not complete until this document is updated.
