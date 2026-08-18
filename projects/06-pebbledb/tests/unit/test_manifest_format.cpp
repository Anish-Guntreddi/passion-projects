#include "pebbledb/manifest/format.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// Pure in-memory encode/decode tests: no filesystem involved -- the
// manifest counterpart of tests/unit/test_wal_record_codec.cpp and
// tests/unit/test_sstable_format.cpp. Real on-disk Save/Load behavior
// (write-new + fsync + atomic rename, D7) lives in
// tests/integration/test_db_persistence.cpp; real-file corruption lives
// in tests/corruption/test_manifest_corruption.cpp.

namespace pebbledb::manifest {
namespace {

TEST(ManifestFormat, EmptyManifestRoundTrips) {
  ManifestData data;
  data.next_file_number = 1;
  data.wal_file_number = 0;
  data.last_sequence = 0;

  std::string encoded;
  EncodeManifest(data, &encoded);
  EXPECT_EQ(encoded.size(), kHeaderSize);  // no table entries

  ManifestData decoded;
  ASSERT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.next_file_number, 1u);
  EXPECT_EQ(decoded.wal_file_number, 0u);
  EXPECT_EQ(decoded.last_sequence, 0u);
  EXPECT_TRUE(decoded.tables.empty());
}

TEST(ManifestFormat, ManifestWithTablesRoundTrips) {
  ManifestData data;
  data.next_file_number = 42;
  data.wal_file_number = 17;
  data.last_sequence = 999;
  data.tables = {
      TableFileInfo{2, 0},
      TableFileInfo{5, 0},
      TableFileInfo{9, 0},
  };

  std::string encoded;
  EncodeManifest(data, &encoded);
  EXPECT_EQ(encoded.size(), kHeaderSize + 3 * kTableEntrySize);

  ManifestData decoded;
  ASSERT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.next_file_number, 42u);
  EXPECT_EQ(decoded.wal_file_number, 17u);
  EXPECT_EQ(decoded.last_sequence, 999u);
  ASSERT_EQ(decoded.tables.size(), 3u);
  EXPECT_EQ(decoded.tables[0].file_number, 2u);
  EXPECT_EQ(decoded.tables[1].file_number, 5u);
  EXPECT_EQ(decoded.tables[2].file_number, 9u);
  // Table order is preserved exactly (oldest-flushed-first -- see
  // format.hpp's header comment) -- not sorted, not reversed.
}

// Table entries are stored in whatever order EncodeManifest is given
// (oldest-flushed-first, by DB::FlushLocked()'s convention -- format.hpp)
// -- DecodeManifest must not silently reorder them.
TEST(ManifestFormat, TableOrderIsPreservedNotSorted) {
  ManifestData data;
  data.tables = {TableFileInfo{9, 0}, TableFileInfo{2, 0}, TableFileInfo{5, 0}};
  std::string encoded;
  EncodeManifest(data, &encoded);

  ManifestData decoded;
  ASSERT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kOk);
  ASSERT_EQ(decoded.tables.size(), 3u);
  EXPECT_EQ(decoded.tables[0].file_number, 9u);
  EXPECT_EQ(decoded.tables[1].file_number, 2u);
  EXPECT_EQ(decoded.tables[2].file_number, 5u);
}

TEST(ManifestFormat, NonzeroLevelRoundTrips) {
  // level is always 0 in this project's own writer output (reserved for
  // Phase 6 -- format.hpp), but a reader must not reject a nonzero value
  // outright.
  ManifestData data;
  data.tables = {TableFileInfo{1, 3}};
  std::string encoded;
  EncodeManifest(data, &encoded);

  ManifestData decoded;
  ASSERT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kOk);
  ASSERT_EQ(decoded.tables.size(), 1u);
  EXPECT_EQ(decoded.tables[0].level, 3u);
}

TEST(ManifestFormat, DecodeRejectsTooFewBytesForHeader) {
  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(std::string(kHeaderSize - 1, '\0'), &decoded), DecodeStatus::kTruncated);
}

TEST(ManifestFormat, DecodeRejectsEmptyInput) {
  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(std::string_view(), &decoded), DecodeStatus::kTruncated);
}

TEST(ManifestFormat, DecodeRejectsBadMagic) {
  ManifestData data;
  std::string encoded;
  EncodeManifest(data, &encoded);
  // magic starts right after the 4-byte checksum.
  encoded[4] ^= 0xFF;

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kBadMagic);
}

TEST(ManifestFormat, DecodeRejectsUnsupportedVersion) {
  ManifestData data;
  std::string encoded;
  EncodeManifest(data, &encoded);
  // format_version is the single byte right after the 4-byte magic.
  encoded[8] = static_cast<char>(kFormatVersion + 1);
  // Recompute nothing -- an unsupported version must be caught *before*
  // the checksum is even consulted (format.hpp's documented validation
  // order), so a stale checksum here must not matter.

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kUnsupportedVersion);
}

TEST(ManifestFormat, DecodeRejectsFormatVersionZero) {
  ManifestData data;
  std::string encoded;
  EncodeManifest(data, &encoded);
  encoded[8] = 0;

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kUnsupportedVersion);
}

TEST(ManifestFormat, DecodeRejectsTableCountExceedingSanityBound) {
  // Hand-craft a header whose num_tables field is corrupted to something
  // enormous, without actually supplying that many table entries --
  // exactly the "corrupted length field must fail fast, not drive an
  // unbounded allocation" scenario ADR 0006/ADR 0011 guard against
  // elsewhere in this codebase (invariant 7).
  ManifestData data;
  std::string encoded;
  EncodeManifest(data, &encoded);
  ASSERT_EQ(encoded.size(), kHeaderSize);

  // num_tables is the last 4 bytes of the header.
  const std::size_t num_tables_offset = kHeaderSize - 4;
  const std::uint32_t huge = kMaxTableCount + 1;
  encoded[num_tables_offset + 0] = static_cast<char>(huge & 0xFF);
  encoded[num_tables_offset + 1] = static_cast<char>((huge >> 8) & 0xFF);
  encoded[num_tables_offset + 2] = static_cast<char>((huge >> 16) & 0xFF);
  encoded[num_tables_offset + 3] = static_cast<char>((huge >> 24) & 0xFF);

  ManifestData decoded;
  // The checksum is now stale (num_tables changed after EncodeManifest
  // computed it), but kBadTableCount must be reported *before* the
  // checksum is ever consulted -- see format.hpp's documented validation
  // order ("num_tables against kMaxTableCount... before it is used to
  // compute an expected size, then the buffer's actual size... then
  // finally the checksum").
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kBadTableCount);
}

TEST(ManifestFormat, DecodeRejectsShortBufferForDeclaredTableCount) {
  ManifestData data;
  data.tables = {TableFileInfo{1, 0}, TableFileInfo{2, 0}};
  std::string encoded;
  EncodeManifest(data, &encoded);
  // Truncate away the last table entry's bytes, but keep num_tables == 2.
  encoded.resize(encoded.size() - kTableEntrySize);

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kTruncated);
}

TEST(ManifestFormat, DecodeRejectsTrailingExtraBytes) {
  // The manifest is a whole-file format, not an appendable log (unlike
  // the WAL) -- trailing bytes beyond the last declared table entry are
  // exactly as invalid as a short file (format.hpp's kBadLength doc
  // comment), never "extra data to ignore."
  ManifestData data;
  data.tables = {TableFileInfo{1, 0}};
  std::string encoded;
  EncodeManifest(data, &encoded);
  encoded.push_back('\0');

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kBadLength);
}

TEST(ManifestFormat, DecodeRejectsChecksumMismatch) {
  ManifestData data;
  data.next_file_number = 5;
  data.tables = {TableFileInfo{1, 0}};
  std::string encoded;
  EncodeManifest(data, &encoded);
  // Flip a byte inside the checksummed region (past the header's own
  // fixed fields, inside the one table entry) without touching the
  // structural fields validated first (magic/version/num_tables) or the
  // stored checksum itself.
  encoded[encoded.size() - 1] ^= 0xFF;

  ManifestData decoded;
  EXPECT_EQ(DecodeManifest(encoded, &decoded), DecodeStatus::kChecksumMismatch);
}

TEST(ManifestFormat, DecodeStatusNameCoversEveryEnumerator) {
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kOk), "Ok");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kTruncated), "Truncated");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadMagic), "BadMagic");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kUnsupportedVersion), "UnsupportedVersion");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadTableCount), "BadTableCount");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadLength), "BadLength");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kChecksumMismatch), "ChecksumMismatch");
}

}  // namespace
}  // namespace pebbledb::manifest
