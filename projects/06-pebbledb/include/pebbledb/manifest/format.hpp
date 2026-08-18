#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pebbledb::manifest {

// Manifest v1 binary format (spec FR5; full field-by-field documentation
// lives in docs/file-format.md -- this header is a pointer, not a
// duplicate source of truth). See ADR 0012 (binary format and D7's
// write-new + fsync + atomic-rename atomicity) and ADR 0013 (the flush/
// recovery design this format serves).
//
// Unlike the WAL (an append-only log) or an SSTable (many independent
// blocks), the manifest is always read and written as one single, whole
// file -- there is exactly one on-disk manifest per DB directory, at a
// fixed canonical name (manifest::kManifestFileName), and every update
// atomically replaces its entire contents (never appended to, never
// partially rewritten in place). Every multi-byte field is fixed-width
// and little-endian, and the whole file is protected by one CRC-32C
// checksum -- the same conventions ADR 0007/0010 established for the WAL
// and SSTable formats, reused here per ADR 0012.
//
// Layout, in on-disk order:
//
//   checksum            uint32   CRC-32C over every byte from `magic`
//                                through the last table entry (i.e. the
//                                whole file except this field itself).
//   magic                uint32   constant kMagic.
//   format_version        uint8    constant kFormatVersion (currently 1).
//   next_file_number     uint64   next number to allocate for a new
//                                on-disk file (a WAL segment or an
//                                SSTable) -- see ADR 0013.
//   wal_file_number      uint64   file number of the WAL that must be
//                                replayed, after loading this manifest,
//                                to reconstruct any writes durable in the
//                                WAL but not yet captured in an SSTable.
//   last_sequence        uint64   highest sequence number known durable
//                                as of this manifest snapshot.
//   num_tables           uint32   number of table entries that follow.
//   table entries        num_tables * kTableEntrySize bytes, back to
//                                back -- see TableFileInfo below.
//
// Table entries are stored oldest-flushed-first (the order tables were
// added to the manifest across successive flushes); a reader resolving a
// point lookup must consult them **newest-first** (i.e. iterate the
// stored list in reverse) -- a more-recently-flushed table's entry for a
// key shadows an older table's entry for the same key, exactly the same
// "newest layer wins outright" rule docs/architecture.md's read path
// already applies to MemTableList's active-vs-immutable layers. See ADR
// 0013.

struct TableFileInfo {
  std::uint64_t file_number = 0;
  // Always 0 in this version -- reserved for Phase 6 compaction's
  // leveling, exactly as ADR 0009 reserved the SSTable footer's
  // filter_handle field for Phase 5's Bloom filter before Phase 5 existed.
  // A reader must not reject a nonzero value outright (a future writer
  // populating it is not a format-version bump), but nothing in this
  // version's DB ever produces or consults a nonzero level.
  std::uint32_t level = 0;
};

struct ManifestData {
  std::uint64_t next_file_number = 1;
  std::uint64_t wal_file_number = 0;
  std::uint64_t last_sequence = 0;
  // Oldest-flushed-first -- see this file's header comment.
  std::vector<TableFileInfo> tables;
};

inline constexpr std::uint32_t kMagic = 0x50424D46u;  // "PBMF", see docs/file-format.md
inline constexpr std::uint8_t kFormatVersion = 1;

// Fixed canonical on-disk name of the manifest file, and of the
// temporary file `manifest::SaveManifest` (manifest.hpp) writes-then-
// atomically-renames over it (spec decision D7; ADR 0012). Both live
// directly under the DB directory, alongside the numbered WAL/SSTable
// files named by util::WalFileName/SstableFileName (util/filenames.hpp)
// -- the manifest is deliberately NOT part of that numbered scheme,
// since exactly one manifest exists per DB directory at all times.
inline constexpr const char* kManifestFileName = "MANIFEST";
inline constexpr const char* kManifestTempFileName = "MANIFEST.tmp";

// checksum(4) + magic(4) + format_version(1) + next_file_number(8) +
// wal_file_number(8) + last_sequence(8) + num_tables(4)
inline constexpr std::size_t kHeaderSize = 4 + 4 + 1 + 8 + 8 + 8 + 4;  // 37

// file_number(8) + level(4)
inline constexpr std::size_t kTableEntrySize = 8 + 4;  // 12

// Sanity bound on the declared `num_tables` field, checked before it is
// ever used to compute an expected total file size or reserve a vector
// -- the same "fail fast on a corrupted length field instead of driving
// an unbounded allocation" discipline as wal::kMaxKeyLength/
// kMaxValueLength (ADR 0006) and sstable::Reader's BlockHandle bounds
// checks (ADR 0011). Generous relative to any table count this project's
// synchronous, non-leveled Phase 4/5 design ever produces.
inline constexpr std::uint32_t kMaxTableCount = 1u << 20;  // ~1,048,576

enum class DecodeStatus {
  kOk,
  kTruncated,           // fewer bytes supplied than the structure needs
  kBadMagic,
  kUnsupportedVersion,
  kBadTableCount,       // num_tables exceeds kMaxTableCount
  kBadLength,           // buffer size does not match num_tables exactly
                        // (too short *or* too long -- the manifest is a
                        // whole-file format, not an appendable log, so
                        // trailing extra bytes are exactly as invalid as
                        // a torn/short file; see ADR 0012)
  kChecksumMismatch,
};

const char* DecodeStatusName(DecodeStatus status);

// Serializes `data` in the v1 on-disk format described above, appending
// the bytes to `dst` (not cleared first).
void EncodeManifest(const ManifestData& data, std::string* dst);

// Decodes and validates `bytes`, which must be exactly one complete,
// whole manifest file's contents (not a stream/cursor API, unlike the
// WAL's DecodeHeader/DecodeBody split -- the manifest is always read
// whole into memory first, the same way sstable::Reader::Open reads the
// whole fixed-size footer at once, generalized to a variable-but-
// self-describing total length here). Validation order: magic, then
// format_version, then num_tables against kMaxTableCount (before it is
// used to compute an expected size), then the buffer's actual size
// against that expected size, then finally the checksum (which needs the
// now-trusted-length buffer fully in hand) -- mirroring the WAL/SSTable
// formats' "structural fields before checksum" convention, adapted to a
// format whose checksum coverage length itself depends on a
// not-yet-checksum-verified field (num_tables), the same bootstrapping
// problem the WAL's key_length/value_length already solve the same way
// (ADR 0006/0007).
DecodeStatus DecodeManifest(std::string_view bytes, ManifestData* out);

}  // namespace pebbledb::manifest
