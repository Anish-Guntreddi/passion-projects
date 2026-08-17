#include "pebbledb/sstable/format.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// Pure in-memory encode/decode tests: no filesystem involved -- the
// sstable counterpart of tests/unit/test_wal_record_codec.cpp. Real
// on-disk Writer/Reader behavior (including reopening after "restart")
// lives in tests/integration/test_sstable_writer_reader.cpp; real-file
// corruption lives in tests/corruption/test_sstable_corruption.cpp.

namespace pebbledb::sstable {
namespace {

// --- Entry ---------------------------------------------------------------

TEST(SstableFormatEntry, ValueEntryRoundTrips) {
  std::string encoded;
  EncodeEntry("key1", 42, EntryType::kValue, "value1", &encoded);

  std::string_view view = encoded;
  Entry decoded;
  ASSERT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.key, "key1");
  EXPECT_EQ(decoded.sequence, 42u);
  EXPECT_EQ(decoded.type, EntryType::kValue);
  EXPECT_EQ(decoded.value, "value1");
  EXPECT_TRUE(view.empty());  // fully consumed
}

TEST(SstableFormatEntry, TombstoneEntryIgnoresSuppliedValue) {
  std::string encoded;
  EncodeEntry("gone", 7, EntryType::kTombstone, "this-must-never-be-persisted", &encoded);

  std::string_view view = encoded;
  Entry decoded;
  ASSERT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.type, EntryType::kTombstone);
  EXPECT_EQ(decoded.value, "");
  // The encoding itself must be exactly kEntryHeaderSize + key size (no
  // value bytes at all), not merely "decodes to an empty value".
  EXPECT_EQ(encoded.size(), kEntryHeaderSize + 4);
}

TEST(SstableFormatEntry, EmptyKeyAndEmptyValueRoundTrip) {
  std::string encoded;
  EncodeEntry("", 1, EntryType::kValue, "", &encoded);
  std::string_view view = encoded;
  Entry decoded;
  ASSERT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.key, "");
  EXPECT_EQ(decoded.value, "");
}

TEST(SstableFormatEntry, BinarySafeKeyAndValueRoundTrip) {
  std::string key("k\0ey", 4);
  std::string value;
  for (int b = 0; b < 256; ++b) {
    value.push_back(static_cast<char>(static_cast<unsigned char>(b)));
  }
  std::string encoded;
  EncodeEntry(key, 5, EntryType::kValue, value, &encoded);

  std::string_view view = encoded;
  Entry decoded;
  ASSERT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.key, key);
  EXPECT_EQ(decoded.value, value);
}

TEST(SstableFormatEntry, SequenceRoundTripsAtUint64Max) {
  std::string encoded;
  EncodeEntry("k", std::numeric_limits<std::uint64_t>::max(), EntryType::kValue, "v", &encoded);
  std::string_view view = encoded;
  Entry decoded;
  ASSERT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.sequence, std::numeric_limits<std::uint64_t>::max());
}

TEST(SstableFormatEntry, MultipleEntriesConcatenateAndDecodeInOrder) {
  std::string buf;
  EncodeEntry("a", 1, EntryType::kValue, "1", &buf);
  EncodeEntry("b", 2, EntryType::kTombstone, "", &buf);
  EncodeEntry("c", 3, EntryType::kValue, "3", &buf);

  std::string_view remaining = buf;
  std::vector<std::string> keys;
  while (!remaining.empty()) {
    Entry entry;
    ASSERT_EQ(DecodeEntry(&remaining, &entry), DecodeStatus::kOk);
    keys.push_back(entry.key);
  }
  ASSERT_EQ(keys.size(), 3u);
  EXPECT_EQ(keys[0], "a");
  EXPECT_EQ(keys[1], "b");
  EXPECT_EQ(keys[2], "c");
}

TEST(SstableFormatEntry, DecodeRejectsTruncatedHeader) {
  std::string encoded;
  EncodeEntry("key", 1, EntryType::kValue, "value", &encoded);

  for (std::size_t len = 0; len < kEntryHeaderSize; ++len) {
    std::string_view view = std::string_view(encoded).substr(0, len);
    Entry decoded;
    EXPECT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kTruncated) << "len=" << len;
  }
}

TEST(SstableFormatEntry, DecodeRejectsTruncatedPayload) {
  std::string encoded;
  EncodeEntry("key", 1, EntryType::kValue, "value12345", &encoded);

  for (std::size_t len = kEntryHeaderSize; len < encoded.size(); ++len) {
    std::string_view view = std::string_view(encoded).substr(0, len);
    Entry decoded;
    EXPECT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kTruncated) << "len=" << len;
  }
}

TEST(SstableFormatEntry, DecodeRejectsBadEntryType) {
  std::string encoded;
  EncodeEntry("key", 1, EntryType::kValue, "value", &encoded);
  encoded[0] = static_cast<char>(0);  // entry_type byte, offset 0: neither kValue nor kTombstone

  std::string_view view = encoded;
  Entry decoded;
  EXPECT_EQ(DecodeEntry(&view, &decoded), DecodeStatus::kBadEntryType);
}

// --- IndexEntry ------------------------------------------------------------

TEST(SstableFormatIndexEntry, RoundTrips) {
  IndexEntry entry;
  entry.last_key = "zzz";
  entry.handle.offset = 4096;
  entry.handle.size = 128;

  std::string encoded;
  EncodeIndexEntry(entry, &encoded);

  std::string_view view = encoded;
  IndexEntry decoded;
  ASSERT_EQ(DecodeIndexEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.last_key, "zzz");
  EXPECT_EQ(decoded.handle.offset, 4096u);
  EXPECT_EQ(decoded.handle.size, 128u);
  EXPECT_TRUE(view.empty());
}

TEST(SstableFormatIndexEntry, EmptyKeyRoundTrips) {
  IndexEntry entry;
  entry.last_key = "";
  entry.handle.offset = 0;
  entry.handle.size = 5;

  std::string encoded;
  EncodeIndexEntry(entry, &encoded);
  std::string_view view = encoded;
  IndexEntry decoded;
  ASSERT_EQ(DecodeIndexEntry(&view, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.last_key, "");
}

TEST(SstableFormatIndexEntry, MultipleEntriesConcatenateAndDecodeInOrder) {
  std::string buf;
  IndexEntry a{"a", BlockHandle{0, 10}};
  IndexEntry b{"b", BlockHandle{10, 20}};
  EncodeIndexEntry(a, &buf);
  EncodeIndexEntry(b, &buf);

  std::string_view remaining = buf;
  IndexEntry decoded_a, decoded_b;
  ASSERT_EQ(DecodeIndexEntry(&remaining, &decoded_a), DecodeStatus::kOk);
  ASSERT_EQ(DecodeIndexEntry(&remaining, &decoded_b), DecodeStatus::kOk);
  EXPECT_TRUE(remaining.empty());
  EXPECT_EQ(decoded_a.last_key, "a");
  EXPECT_EQ(decoded_b.last_key, "b");
  EXPECT_EQ(decoded_b.handle.offset, 10u);
}

TEST(SstableFormatIndexEntry, DecodeRejectsTruncated) {
  IndexEntry entry{"somekey", BlockHandle{1, 2}};
  std::string encoded;
  EncodeIndexEntry(entry, &encoded);

  for (std::size_t len = 0; len < encoded.size(); ++len) {
    std::string_view view = std::string_view(encoded).substr(0, len);
    IndexEntry decoded;
    EXPECT_EQ(DecodeIndexEntry(&view, &decoded), DecodeStatus::kTruncated) << "len=" << len;
  }
}

// --- Block trailer (FinishBlock / VerifyAndStripBlockTrailer) -------------

TEST(SstableFormatBlock, FinishAndVerifyRoundTrip) {
  std::string contents;
  EncodeEntry("a", 1, EntryType::kValue, "1", &contents);
  EncodeEntry("b", 2, EntryType::kValue, "2", &contents);

  std::string block_bytes;
  std::size_t written = FinishBlock(contents, &block_bytes);
  EXPECT_EQ(written, contents.size() + kBlockTrailerSize);
  EXPECT_EQ(block_bytes.size(), written);

  std::string_view entries_view;
  ASSERT_EQ(VerifyAndStripBlockTrailer(block_bytes, &entries_view), DecodeStatus::kOk);
  EXPECT_EQ(entries_view, std::string_view(contents));
}

TEST(SstableFormatBlock, EmptyBlockRoundTrips) {
  std::string block_bytes;
  std::size_t written = FinishBlock("", &block_bytes);
  EXPECT_EQ(written, kBlockTrailerSize);

  std::string_view entries_view;
  ASSERT_EQ(VerifyAndStripBlockTrailer(block_bytes, &entries_view), DecodeStatus::kOk);
  EXPECT_TRUE(entries_view.empty());
}

TEST(SstableFormatBlock, VerifyRejectsTruncatedBlock) {
  std::string contents = "some entry bytes";
  std::string block_bytes;
  FinishBlock(contents, &block_bytes);

  for (std::size_t len = 0; len < kBlockTrailerSize; ++len) {
    std::string_view truncated = std::string_view(block_bytes).substr(0, len);
    std::string_view entries_view;
    EXPECT_EQ(VerifyAndStripBlockTrailer(truncated, &entries_view), DecodeStatus::kTruncated)
        << "len=" << len;
  }
}

TEST(SstableFormatBlock, VerifyRejectsCorruptedChecksum) {
  std::string contents = "some entry bytes";
  std::string block_bytes;
  FinishBlock(contents, &block_bytes);
  block_bytes[0] ^= 0xFF;  // flip a bit inside the entries region

  std::string_view entries_view;
  EXPECT_EQ(VerifyAndStripBlockTrailer(block_bytes, &entries_view), DecodeStatus::kChecksumMismatch);
}

TEST(SstableFormatBlock, VerifyRejectsCorruptedChecksumFieldItself) {
  std::string contents = "some entry bytes";
  std::string block_bytes;
  FinishBlock(contents, &block_bytes);
  block_bytes.back() ^= 0xFF;  // last byte of the checksum field

  std::string_view entries_view;
  EXPECT_EQ(VerifyAndStripBlockTrailer(block_bytes, &entries_view), DecodeStatus::kChecksumMismatch);
}

TEST(SstableFormatBlock, VerifyRejectsUnsupportedCompressionByte) {
  std::string contents = "some entry bytes";
  std::string block_bytes;
  FinishBlock(contents, &block_bytes);
  // compression_type byte is the first byte of the trailer, i.e. at
  // offset contents.size().
  block_bytes[contents.size()] = static_cast<char>(1);  // not CompressionType::kNone

  std::string_view entries_view;
  EXPECT_EQ(VerifyAndStripBlockTrailer(block_bytes, &entries_view),
            DecodeStatus::kUnsupportedCompression);
}

// --- Footer ----------------------------------------------------------------

Footer MakeSampleFooter() {
  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = BlockHandle{100, 50};
  footer.filter_handle = BlockHandle{0, 0};
  footer.num_entries = 12345;
  return footer;
}

TEST(SstableFormatFooter, RoundTrips) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  EXPECT_EQ(encoded.size(), kFooterSize);

  Footer decoded;
  ASSERT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kOk);
  EXPECT_EQ(decoded.format_version, footer.format_version);
  EXPECT_EQ(decoded.index_handle.offset, footer.index_handle.offset);
  EXPECT_EQ(decoded.index_handle.size, footer.index_handle.size);
  EXPECT_EQ(decoded.filter_handle.offset, footer.filter_handle.offset);
  EXPECT_EQ(decoded.filter_handle.size, footer.filter_handle.size);
  EXPECT_EQ(decoded.num_entries, footer.num_entries);
}

TEST(SstableFormatFooter, DecodeRejectsTruncated) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);

  for (std::size_t len = 0; len < kFooterSize; ++len) {
    Footer decoded;
    EXPECT_EQ(DecodeFooter(std::string_view(encoded).substr(0, len), &decoded),
              DecodeStatus::kTruncated)
        << "len=" << len;
  }
}

TEST(SstableFormatFooter, DecodeRejectsBadMagic) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  encoded[4] ^= 0xFF;  // magic occupies bytes [4, 8)

  Footer decoded;
  EXPECT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kBadMagic);
}

TEST(SstableFormatFooter, DecodeRejectsUnsupportedNewerVersion) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  encoded[8] = static_cast<char>(kFormatVersion + 1);  // format_version byte, offset 8
  // Deliberately not fixing up the checksum: version is checked before
  // the checksum is even attempted, mirroring wal::DecodeHeader's
  // ordering rationale (docs/file-format.md).

  Footer decoded;
  EXPECT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kUnsupportedVersion);
}

TEST(SstableFormatFooter, DecodeRejectsZeroVersion) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  encoded[8] = 0;

  Footer decoded;
  EXPECT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kUnsupportedVersion);
}

TEST(SstableFormatFooter, DecodeRejectsChecksumMismatch) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  // Corrupt a byte inside the checksummed region (num_entries field,
  // near the end) without touching magic/version.
  encoded.back() ^= 0xFF;

  Footer decoded;
  EXPECT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kChecksumMismatch);
}

TEST(SstableFormatFooter, DecodeRejectsDirectlyCorruptedChecksumField) {
  Footer footer = MakeSampleFooter();
  std::string encoded;
  EncodeFooter(footer, &encoded);
  encoded[0] ^= 0xFF;  // checksum field itself, offset 0

  Footer decoded;
  EXPECT_EQ(DecodeFooter(encoded, &decoded), DecodeStatus::kChecksumMismatch);
}

TEST(SstableFormat, DecodeStatusNameCoversEveryEnumerator) {
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kOk), "Ok");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kTruncated), "Truncated");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadMagic), "BadMagic");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kUnsupportedVersion), "UnsupportedVersion");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kChecksumMismatch), "ChecksumMismatch");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadEntryType), "BadEntryType");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kUnsupportedCompression), "UnsupportedCompression");
}

}  // namespace
}  // namespace pebbledb::sstable
