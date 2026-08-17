# ADR 0007: WAL record v1 binary format — encoding choices

- **Status:** Accepted
- **Phase:** 1 (WAL)
- **Spec requirement:** FR10 — "All binary formats versioned from the
  beginning... WAL record header fields: magic/version, operation, sequence
  number, key length, value length, checksum (exact encoding is a design
  decision → ADR)."

This ADR covers the *encoding* choices behind the WAL v1 record format —
byte order, integer width, and checksum algorithm/coverage. The field-by-
field on-disk layout table is the single source of truth in
`docs/file-format.md`; this ADR explains *why* those choices were made,
which `docs/file-format.md` deliberately does not re-litigate. See ADR 0005
for the sequence-number field specifically, and ADR 0006 for the sync-
policy and resync-on-corruption semantics around this format.

## Decision — fixed-width, little-endian integers

Every multi-byte field (`checksum`, `magic`, `sequence`, `key_length`,
`value_length`) is encoded as a fixed-width, explicitly little-endian
integer via `util::PutFixed32`/`PutFixed64` (`include/pebbledb/util/
coding.hpp`), which shift bytes out/in one at a time rather than doing a raw
`memcpy`/`reinterpret_cast` of a native integer.

- **Fixed-width, not varint:** every WAL record has the same 26-byte header
  regardless of field values, so `kHeaderSize` is a compile-time constant
  and header parsing never needs to branch on how many bytes a given field
  consumed. Varints would save bytes for small values (most sequence
  numbers, most key/value lengths) but would make `kHeaderSize` variable
  and complicate both `DecodeHeader` and the "read the header, then read
  exactly `key_length + value_length` more bytes" framing `wal::Reader`
  relies on. At this project's scope, the byte savings are not worth that
  complexity.
- **Little-endian, chosen arbitrarily** (there is no correctness reason to
  prefer it over big-endian for this project — it matches the native byte
  order of essentially every platform this code runs or is developed on,
  which is a minor implementation convenience, not a design requirement).
  What matters is that it is *fixed and explicit*, not "whatever the host's
  native order happens to be": encoding and decoding always go through
  `PutFixed32`/`PutFixed64`/`DecodeFixed32`/`DecodeFixed64`, never a direct
  struct-cast of the buffer, so the on-disk format is portable across hosts
  regardless of their native endianness and does not depend on undefined
  behavior (strict-aliasing violations a `reinterpret_cast` of a packed
  struct would risk).

## Decision — CRC-32C checksum

- **Algorithm:** CRC-32C, the Castagnoli polynomial (`0x1EDC6F41`, reversed-
  bit form `0x82F63B78`), implemented as a from-scratch, table-based
  ("slice-by-1") software CRC (`src/util/crc32c.cpp`) — not vendored from an
  external library. The 256-entry lookup table is generated once via
  `constexpr` evaluation; this is the standard, well-documented software CRC
  construction.
- **Why CRC-32C over classic CRC-32** (the zlib/gzip polynomial,
  `0xEDB88320`): CRC-32C has strictly better error-detection properties
  (Hamming distance) at the short record sizes a WAL record or SSTable block
  actually uses, and is the same polynomial already used for exactly this
  purpose elsewhere (iSCSI, SCTP, ext4 metadata, and the SSE4.2 hardware
  `CRC32` instruction all use CRC-32C) — a well-trodden, appropriate choice
  for this exact use case rather than a novel one.
- **Checksum coverage:** every byte of the record *after* the checksum field
  itself — `magic` through the end of `value` — computed in one pass over
  the concatenation (`EncodeRecord`), or equivalently via `Crc32c(header)`
  followed by `ExtendCrc32c(crc, key.data(), key.size())` then
  `ExtendCrc32c(crc, value.data(), value.size())` (`DecodeBody`), which lets
  the checksum be verified across the non-contiguous header/payload buffers
  a `Reader` actually has in hand without concatenating them first. The
  checksum field itself is excluded from its own coverage for the obvious
  reason (a field cannot check itself).
- **Verification order relative to other validation:** `DecodeHeader`
  validates `magic`, `format_version`, and the length sanity bounds
  (`kMaxKeyLength`/`kMaxValueLength`) *before* the checksum is ever computed
  (the checksum needs the payload too, which isn't available yet at header-
  decode time). `DecodeBody` verifies the checksum *before* validating the
  operation byte (`kPut`/`kDelete`) — deliberately, so a corrupted op byte is
  reported consistently as `kChecksumMismatch` (the checksum will not match
  once any byte in its coverage, including the op byte, is corrupted)
  rather than racing between "checksum failure" and "bad operation" based on
  which corrupted byte happens to be checked first. See
  `tests/unit/test_wal_record_codec.cpp`'s
  `DecodeBodyRejectsInvalidOperationByte` test, which hand-crafts a record
  with a *correct* checksum over an invalid op byte specifically to isolate
  this ordering: a self-consistent-but-invalid record must still be
  rejected on the operation-byte check once checksum verification passes.

## Consequences

- The 26-byte fixed header (`checksum` 4 + `magic` 4 + `format_version` 1 +
  `op` 1 + `sequence` 8 + `key_length` 4 + `value_length` 4) is a fixed,
  known cost per record, documented in `docs/file-format.md`; it does not
  shrink for small keys/values the way a varint-encoded format would.
- Any change to byte order, checksum algorithm, or checksum coverage is a
  new format version (`format_version` bump), a `docs/file-format.md`
  update, and a new ADR superseding the relevant part of this one — never a
  silent reinterpretation of v1's bytes. `DecodeHeader` already rejects any
  `format_version` other than exactly `kFormatVersion` (0 or newer-than-
  understood are both `kUnsupportedVersion`), so this is enforced by the
  reader, not just documentation.
- SSTable blocks (Phase 3) and the manifest (Phase 4) are expected to reuse
  the same fixed-width/little-endian/CRC-32C conventions established here
  for consistency, documented in their own ADRs and `docs/file-format.md`
  sections when those phases begin — this ADR is WAL-v1-specific but the
  conventions it establishes are the project's general-purpose byte-
  encoding toolkit (`util::coding`, `util::crc32c`), already written to be
  reused rather than WAL-specific.
