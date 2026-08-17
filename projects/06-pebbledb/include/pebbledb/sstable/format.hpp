#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "pebbledb/entry_type.hpp"

namespace pebbledb::sstable {

// SSTable v1 binary format (spec FR3; full field-by-field documentation
// lives in docs/file-format.md -- this header is a pointer, not a
// duplicate source of truth). See ADR 0009 (block size / compression,
// spec decision D5) and ADR 0010 (encoding rationale, the SSTable
// counterpart of ADR 0007's role for the WAL format).
//
// File layout, in on-disk order:
//
//   [data block 0] [data block 1] ... [data block N-1]
//   [filter block]   (reserved for Phase 5's Bloom filter; zero-length in
//                      this version's writer output -- see ADR 0009)
//   [index block]
//   [footer]          (fixed kFooterSize bytes, always the file's last
//                      kFooterSize bytes -- a reader locates everything
//                      else by seeking there first)
//
// Every multi-byte field is fixed-width and little-endian
// (util::PutFixed32/64), exactly like the WAL format (ADR 0007's
// conventions, reused here per ADR 0010).

inline constexpr std::uint32_t kMagic = 0x50425354u;  // "PBST", see docs/file-format.md
inline constexpr std::uint8_t kFormatVersion = 1;

// Every entry within a data block (or the index block -- see IndexEntry
// below for that block's own, different entry shape) is a fixed header
// immediately followed by the variable-length key/value payload:
//
//   entry_type    uint8    EntryType::kValue (1) or kTombstone (2)
//   sequence      uint64
//   key_length    uint32
//   value_length  uint32   (always 0 for kTombstone)
//   key           key_length bytes
//   value         value_length bytes (absent for kTombstone)
inline constexpr std::size_t kEntryHeaderSize = 1 + 8 + 4 + 4;  // 17

// Appended after a block's raw, concatenated entry (or index-entry) bytes
// to close out a data/index block:
//
//   compression_type   uint8    CompressionType::kNone (0) in this
//                                version -- see CompressionType below.
//   checksum           uint32   CRC-32C over [entries bytes] followed by
//                                [compression_type byte] (i.e. every byte
//                                of the block except this checksum field
//                                itself).
inline constexpr std::size_t kBlockTrailerSize = 1 + 4;  // 5

enum class CompressionType : std::uint8_t {
  kNone = 0,
  // Snappy/Zstd block compression is a documented stretch goal (spec Part
  // 3, "Stretch goals"), not implemented here. kNone is the only value
  // this version's Writer ever emits and the only value this version's
  // Reader accepts -- anything else decodes as
  // DecodeStatus::kUnsupportedCompression, reported distinctly from
  // corruption (mirrors wal::ReadResult::kUnsupportedVersion's rationale:
  // an unrecognized-but-structurally-valid value is not evidence of bit
  // rot).
};

// The location and size of one block within the file. Used both for data
// blocks (recorded per-block in the index block) and for the index/filter
// blocks themselves (recorded in the footer). `size` is the block's total
// on-disk size -- entries plus kBlockTrailerSize -- i.e. exactly how many
// bytes a reader must pread starting at `offset` to get the whole block.
struct BlockHandle {
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
};

// One entry in the index block: the last (largest) key present in a data
// block, paired with that block's handle. Data blocks are non-overlapping
// and written in increasing key order, so "last key in block N" is a
// valid separator: the first index entry whose last_key >= a target key
// identifies the one data block that could contain it (see
// sstable::Reader::Get).
//
//   key_length    uint32
//   last_key      key_length bytes
//   block_offset  uint64
//   block_size    uint64
struct IndexEntry {
  std::string last_key;
  BlockHandle handle;
};

// checksum(4) + magic(4) + format_version(1) + index_handle(16) +
// filter_handle(16) + num_entries(8)
inline constexpr std::size_t kFooterSize = 4 + 4 + 1 + 16 + 16 + 8;  // 49

struct Footer {
  std::uint32_t stored_checksum = 0;
  std::uint8_t format_version = 0;
  BlockHandle index_handle;
  // {0, 0} means "no filter block present" -- always true in this
  // version's writer output; Phase 5 populates this. A reader must treat
  // size == 0 as "absent, do not attempt to read it", not as an error.
  BlockHandle filter_handle;
  std::uint64_t num_entries = 0;
};

enum class DecodeStatus {
  kOk,
  kTruncated,               // fewer bytes supplied than the structure needs
  kBadMagic,
  kUnsupportedVersion,
  kChecksumMismatch,
  kBadEntryType,
  kUnsupportedCompression,
};

const char* DecodeStatusName(DecodeStatus status);

// A single decoded data-block entry (see kEntryHeaderSize's layout above).
struct Entry {
  std::string key;
  std::uint64_t sequence = 0;
  EntryType type = EntryType::kValue;
  std::string value;  // empty/meaningless for kTombstone
};

// Appends one entry's on-disk encoding to `dst`. `type == kTombstone`
// ignores `value` and encodes value_length as 0, exactly like
// wal::EncodeRecord does for a kDelete record.
void EncodeEntry(const std::string& key, std::uint64_t sequence, EntryType type,
                  const std::string& value, std::string* dst);

// Decodes exactly one entry starting at the front of `*input`. On
// success, advances `*input` past the bytes consumed (so callers can loop
// `while (!input.empty()) { DecodeEntry(&input, &entry); ... }`) and
// returns kOk. On any failure, `*input` is left unspecified and a
// non-kOk status is returned; callers must stop looping in that case.
//
// No separate "declared length exceeds a sanity bound" check exists here
// (contrast wal::DecodeHeader's kMaxKeyLength/kMaxValueLength): unlike the
// WAL reader, which reads a fixed header off disk and then must decide,
// from a possibly-corrupted length field alone, how many *additional*
// bytes to read from disk, an sstable block is read as one fixed-size,
// already-bounded in-memory buffer (via its BlockHandle::size) before any
// entry is decoded at all. A corrupted key_length/value_length here can
// only ever cause `total > input->size()`, which is already caught below
// and reported as kTruncated -- there is no unbounded-disk-read or
// unbounded-allocation risk to guard against separately.
DecodeStatus DecodeEntry(std::string_view* input, Entry* out);

// Appends `entry`'s on-disk encoding to `dst`.
void EncodeIndexEntry(const IndexEntry& entry, std::string* dst);

// Decodes exactly one index entry starting at the front of `*input`,
// advancing it past the bytes consumed on success. Same conventions as
// DecodeEntry above.
DecodeStatus DecodeIndexEntry(std::string_view* input, IndexEntry* out);

// Wraps `block_contents` (the concatenation of one or more EncodeEntry/
// EncodeIndexEntry calls) with a kNone-compression trailer, appending the
// result to `dst`. Returns the number of bytes appended
// (== block_contents.size() + kBlockTrailerSize) -- exactly the value a
// caller should record as the corresponding BlockHandle::size.
std::size_t FinishBlock(const std::string& block_contents, std::string* dst);

// Verifies `raw_block` -- which must be exactly a BlockHandle::size-byte
// region read from disk, trailer included -- checking the compression
// type and the CRC-32C checksum (invariant 7: corruption must be
// detected, never silently ignored). On success, `*entries_view` is set
// to the entries-only portion (raw_block with the trailer stripped) and
// remains valid only as long as `raw_block`'s backing storage is not
// destroyed or reallocated.
DecodeStatus VerifyAndStripBlockTrailer(std::string_view raw_block, std::string_view* entries_view);

// Encodes `footer` to exactly kFooterSize bytes, appended to `dst`.
void EncodeFooter(const Footer& footer, std::string* dst);

// Decodes and validates a footer from `footer_bytes` (must be exactly
// kFooterSize bytes; fewer is kTruncated). Checks magic, then
// format_version, then the checksum, in that order.
DecodeStatus DecodeFooter(std::string_view footer_bytes, Footer* out);

}  // namespace pebbledb::sstable
