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

## SSTable, manifest formats

Not yet implemented (Phases 3–4 per the roadmap). This section is added,
with the same field-by-field table format as above, in the same change that
introduces each format — per spec Part 4 rule 2, a binary-format task is not
complete until this document is updated.
