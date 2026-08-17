#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pebbledb/sstable/format.hpp"
#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/sstable/writer.hpp"
#include "pebbledb/util/coding.hpp"
#include "temp_file.hpp"

// Invariant 7 (spec §1.5): "Checksums detect corrupted/truncated records
// rather than silently returning garbage." These tests write real
// SSTable files through sstable::Writer, corrupt real bytes on disk via
// testutil::CorruptByteAt/TruncateFileTo, and confirm sstable::Reader
// detects it rather than fabricating (or silently omitting) an entry.
// Pure in-memory decode-path corruption coverage lives in
// tests/unit/test_sstable_format.cpp; this file is the end-to-end,
// real-file version of the same invariant, mirroring
// tests/corruption/test_wal_corruption.cpp's structure for the WAL.

namespace pebbledb::sstable {
namespace {

using pebbledb::testutil::CorruptByteAt;
using pebbledb::testutil::TempFile;

// Writes `count` sorted, distinct keys with small target blocks (forcing
// multiple data blocks), and returns the file's total size for corruption
// tests that need to target byte offsets relative to particular regions.
void WriteMultiBlockTable(const std::string& path, int count) {
  Writer::Options options;
  options.target_block_size_bytes = 48;  // small: forces several data blocks

  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(path, options, &writer).ok());
  for (int i = 0; i < count; ++i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "key%05d", i);
    ASSERT_TRUE(writer->Add(buf, "value" + std::to_string(i), static_cast<std::uint64_t>(i + 1),
                             EntryType::kValue)
                    .ok());
  }
  ASSERT_TRUE(writer->Finish().ok());
}

TEST(SstableCorruption, Invariant7_CorruptingFooterChecksumFailsOpen) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 10);

  std::int64_t size = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size, 0);
  // The footer's checksum field is its first 4 bytes, i.e. the last
  // kFooterSize bytes of the file.
  CorruptByteAt(file.path(), size - static_cast<std::int64_t>(kFooterSize));

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

TEST(SstableCorruption, Invariant7_CorruptingFooterMagicFailsOpen) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 10);

  std::int64_t size = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size, 0);
  // magic occupies bytes [4, 8) of the footer.
  CorruptByteAt(file.path(), size - static_cast<std::int64_t>(kFooterSize) + 4);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

TEST(SstableCorruption, Invariant7_FutureFormatVersionReportedAsNotSupportedNotCorruption) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 5);

  std::int64_t size = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size, 0);
  std::string contents = pebbledb::testutil::ReadWholeFile(file.path());
  // format_version byte is at offset 8 of the footer. Deliberately not
  // fixing up the checksum -- version is checked before the checksum, so
  // this must be reported as NotSupported regardless.
  std::size_t version_offset = contents.size() - kFooterSize + 8;
  contents[version_offset] = static_cast<char>(kFormatVersion + 1);
  pebbledb::testutil::WriteWholeFile(file.path(), contents);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
}

TEST(SstableCorruption, Invariant7_TruncatedFileSmallerThanFooterFailsOpen) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 5);

  pebbledb::testutil::TruncateFileTo(file.path(), static_cast<std::int64_t>(kFooterSize) - 1);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

TEST(SstableCorruption, Invariant7_TruncatedFileMissingIndexBlockFailsOpen) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 20);

  std::unique_ptr<Reader> probe_reader;
  ASSERT_TRUE(Reader::Open(file.path(), &probe_reader).ok());
  const BlockHandle index_handle = probe_reader->footer().index_handle;
  probe_reader.reset();

  std::string contents = pebbledb::testutil::ReadWholeFile(file.path());
  ASSERT_GE(contents.size(), kFooterSize);
  const std::string footer_bytes = contents.substr(contents.size() - kFooterSize);

  // Cut the file off exactly where the index block starts (removing the
  // index block and everything after it up to the old footer), then
  // reattach the *original* footer bytes right after the last data
  // block. The footer itself is therefore still fully intact and valid
  // (magic/version/checksum all verify), but it correctly describes an
  // index block that is no longer present in the file.
  const std::string truncated =
      contents.substr(0, static_cast<std::size_t>(index_handle.offset)) + footer_bytes;
  pebbledb::testutil::WriteWholeFile(file.path(), truncated);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// Corrupting one data block's checksum must fail lookups that land in
// that specific block with a real error (never silently kNotFound, and
// never a fabricated value) -- Open() itself must still succeed, since
// the footer and index block are untouched.
TEST(SstableCorruption, Invariant7_CorruptingDataBlockFailsGetForThatBlockWithError) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 20);

  std::unique_ptr<Reader> probe_reader;
  ASSERT_TRUE(Reader::Open(file.path(), &probe_reader).ok());
  ASSERT_GT(probe_reader->index_entry_count(), 1u) << "test requires multiple data blocks";
  // Copy, not reference: probe_reader (and the index_ vector it owns) is
  // destroyed below, before first_block is used.
  const IndexEntry first_block = probe_reader->index_entries().front();
  probe_reader.reset();

  // Flip a byte inside the first data block's entries region (not its
  // trailer -- any byte before block_size - kBlockTrailerSize works).
  ASSERT_GT(first_block.handle.size, kBlockTrailerSize);
  CorruptByteAt(file.path(), static_cast<std::int64_t>(first_block.handle.offset));

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());  // footer/index still fine

  LookupResult result;
  std::string value;
  Status s = reader->Get(first_block.last_key, &result, &value);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// The other half of the same scenario: a corrupted data block must not
// affect lookups resolved by other, uncorrupted blocks -- sstable::Reader
// has no single "exhausted" cursor the way wal::Reader does (see
// reader.hpp's class comment).
TEST(SstableCorruption, OtherBlocksRemainReadableAfterOneBlockIsCorrupted) {
  TempFile file;
  WriteMultiBlockTable(file.path(), 20);

  std::unique_ptr<Reader> probe_reader;
  ASSERT_TRUE(Reader::Open(file.path(), &probe_reader).ok());
  ASSERT_GT(probe_reader->index_entry_count(), 2u) << "test requires several data blocks";
  const IndexEntry first_block = probe_reader->index_entries().front();
  const IndexEntry last_block = probe_reader->index_entries().back();
  probe_reader.reset();

  CorruptByteAt(file.path(), static_cast<std::int64_t>(first_block.handle.offset));

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());

  // The corrupted block's own key still fails.
  LookupResult result;
  std::string value;
  EXPECT_FALSE(reader->Get(first_block.last_key, &result, &value).ok());

  // A key resolved by the last (uncorrupted) block still succeeds.
  ASSERT_TRUE(reader->Get(last_block.last_key, &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kFound);
}

TEST(SstableCorruption, Invariant7_OversizedIndexKeyLengthIsRejectedNotReadAsHugeAllocation) {
  // Hand-craft a minimal file: a bogus index block whose sole "entry" has
  // an absurd declared key_length, followed by a footer pointing at it.
  // Reader::Open must reject this as corruption rather than attempting a
  // multi-gigabyte read/allocation.
  TempFile file;

  std::string index_block_contents;
  util::PutFixed32(&index_block_contents, 0xFFFFFFFFu);  // key_length: absurd
  // (Deliberately no more bytes -- DecodeIndexEntry must fail on the
  // length check against the available buffer before ever trying to read
  // that many key bytes.)

  std::string index_block_bytes;
  FinishBlock(index_block_contents, &index_block_bytes);

  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = BlockHandle{0, index_block_bytes.size()};
  footer.filter_handle = BlockHandle{0, 0};
  footer.num_entries = 0;

  std::string file_contents = index_block_bytes;
  EncodeFooter(footer, &file_contents);
  pebbledb::testutil::WriteWholeFile(file.path(), file_contents);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// Regression coverage (ADR 0011): a footer can have a perfectly valid
// checksum (EncodeFooter computes it from these exact field values --
// this isn't a bit-flip, but the shape a checksum-covered-but-corrupted
// footer would have) while still declaring an index_handle far larger
// than the file itself. Reader::Open must reject this as corruption
// instead of driving a ~200 GiB pread/allocation
// (util::PosixFile::ReadAt allocates its `size` argument up front).
TEST(SstableCorruption, Invariant7_OversizedFooterIndexHandleIsRejectedNotReadAsHugeAllocation) {
  TempFile file;

  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = BlockHandle{0, static_cast<std::uint64_t>(200) * 1024 * 1024 * 1024};
  footer.filter_handle = BlockHandle{0, 0};
  footer.num_entries = 0;

  std::string file_contents;  // the whole file is exactly this footer
  EncodeFooter(footer, &file_contents);
  ASSERT_EQ(file_contents.size(), kFooterSize);
  pebbledb::testutil::WriteWholeFile(file.path(), file_contents);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// Same shape as above, but for filter_handle -- never actually read by
// this version's Reader (always zero-length in real output, ADR 0009),
// but Open() must still reject a corrupted footer that declares a bogus
// non-zero one rather than silently ignoring it.
TEST(SstableCorruption, Invariant7_OversizedFooterFilterHandleIsRejectedAtOpen) {
  TempFile file;

  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = BlockHandle{0, 0};
  footer.filter_handle = BlockHandle{0, static_cast<std::uint64_t>(200) * 1024 * 1024 * 1024};
  footer.num_entries = 0;

  std::string file_contents;
  EncodeFooter(footer, &file_contents);
  ASSERT_EQ(file_contents.size(), kFooterSize);
  pebbledb::testutil::WriteWholeFile(file.path(), file_contents);

  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(file.path(), &reader);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// The per-index-entry counterpart: Open() itself succeeds (the index
// block's own checksum is fine), but the oversized handle it carries for
// one specific data block must still be rejected -- at Get() time, not
// as an unbounded read -- the first time a lookup actually resolves to
// that entry.
TEST(SstableCorruption, Invariant7_OversizedIndexEntryBlockHandleIsRejectedNotReadAsHugeAllocation) {
  TempFile file;

  IndexEntry bogus_entry;
  bogus_entry.last_key = "zzz";
  bogus_entry.handle = BlockHandle{0, static_cast<std::uint64_t>(200) * 1024 * 1024 * 1024};
  std::string index_block_contents;
  EncodeIndexEntry(bogus_entry, &index_block_contents);

  std::string index_block_bytes;
  FinishBlock(index_block_contents, &index_block_bytes);

  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = BlockHandle{0, index_block_bytes.size()};
  footer.filter_handle = BlockHandle{0, 0};
  footer.num_entries = 0;

  std::string file_contents = index_block_bytes;
  EncodeFooter(footer, &file_contents);
  pebbledb::testutil::WriteWholeFile(file.path(), file_contents);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());  // index block itself is intact

  LookupResult result;
  std::string value;
  Status s = reader->Get("zzz", &result, &value);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

}  // namespace
}  // namespace pebbledb::sstable
