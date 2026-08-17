#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pebbledb/util/coding.hpp"
#include "pebbledb/wal/reader.hpp"
#include "pebbledb/wal/writer.hpp"
#include "temp_file.hpp"

// Invariant 7 (spec §1.5): "Checksums detect corrupted/truncated records
// rather than silently returning garbage." These tests write real WAL
// files through wal::Writer, corrupt real bytes on disk via
// testutil::CorruptByteAt, and confirm wal::Reader detects it rather
// than fabricating a record. Pure in-memory decode-path corruption
// coverage lives in tests/unit/test_wal_record_codec.cpp; this file is
// specifically the end-to-end, real-file version of the same invariant.

namespace pebbledb::wal {
namespace {

using pebbledb::testutil::CorruptByteAt;
using pebbledb::testutil::TempFile;

std::vector<Record> WriteSampleWal(const std::string& path, int count) {
  std::vector<Record> records;
  std::unique_ptr<Writer> writer;
  EXPECT_TRUE(Writer::Open(path, SyncPolicy::kEveryWrite, 1, &writer).ok());
  for (int i = 0; i < count; ++i) {
    Record r;
    r.sequence = static_cast<std::uint64_t>(i + 1);
    r.type = RecordType::kPut;
    r.key = "key" + std::to_string(i);
    r.value = "value" + std::to_string(i);
    EXPECT_TRUE(writer->AddRecord(r).ok());
    records.push_back(r);
  }
  return records;
}

bool RecordsEqual(const Record& a, const Record& b) {
  return a.sequence == b.sequence && a.type == b.type && a.key == b.key && a.value == b.value;
}

TEST(WalCorruption, Invariant7_CorruptingFirstRecordStopsReplayAtZeroRecords) {
  TempFile file;
  WriteSampleWal(file.path(), 5);

  // Flip a byte inside the first record's key payload (first record's
  // header occupies bytes [0, kHeaderSize); its key payload starts right
  // after).
  CorruptByteAt(file.path(), static_cast<std::int64_t>(kHeaderSize));

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result = reader->Replay([&](const Record& r) { replayed.push_back(r); });

  EXPECT_EQ(result.terminal, ReadResult::kCorruption);
  EXPECT_EQ(replayed.size(), 0u);
}

TEST(WalCorruption, Invariant7_CorruptingLastRecordPreservesEarlierRecords) {
  TempFile file;
  std::vector<Record> records = WriteSampleWal(file.path(), 5);

  // Corrupt the final byte of the file (inside the last record's value
  // payload).
  std::int64_t size = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size, 0);
  CorruptByteAt(file.path(), size - 1);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result = reader->Replay([&](const Record& r) { replayed.push_back(r); });

  EXPECT_EQ(result.terminal, ReadResult::kCorruption);
  ASSERT_EQ(replayed.size(), 4u);
  for (std::size_t i = 0; i < replayed.size(); ++i) {
    EXPECT_TRUE(RecordsEqual(replayed[i], records[i])) << "record index " << i;
  }
}

TEST(WalCorruption, Invariant7_CorruptingMiddleRecordDoesNotSkipToLaterRecords) {
  TempFile file;
  std::vector<Record> records = WriteSampleWal(file.path(), 5);

  // Locate record 2's (0-indexed) byte range by re-reading record sizes
  // via the checksum-free lengths: simplest robust approach is to
  // corrupt a byte partway through the file and confirm replay stops
  // there rather than "resynchronizing" and picking up records 3-4.
  std::int64_t size = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size, 0);
  CorruptByteAt(file.path(), size / 2);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result = reader->Replay([&](const Record& r) { replayed.push_back(r); });

  // Whatever prefix of complete, uncorrupted records replayed, replay
  // must stop there — it must never contain more records than were
  // written, and it must never "jump past" the corrupted record to
  // recover later ones (no resynchronization).
  EXPECT_LT(replayed.size(), records.size());
  EXPECT_NE(result.terminal, ReadResult::kEndOfLog);
  for (std::size_t i = 0; i < replayed.size(); ++i) {
    EXPECT_TRUE(RecordsEqual(replayed[i], records[i])) << "record index " << i;
  }
}

TEST(WalCorruption, Invariant7_CorruptingChecksumFieldDirectlyIsDetected) {
  TempFile file;
  WriteSampleWal(file.path(), 3);

  // The very first 4 bytes of the file are the first record's checksum
  // field.
  CorruptByteAt(file.path(), 0);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record record;
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kCorruption);
}

TEST(WalCorruption, Invariant7_CorruptingMagicIsDetected) {
  TempFile file;
  WriteSampleWal(file.path(), 3);

  // Magic occupies bytes [4, 8) of the first record.
  CorruptByteAt(file.path(), 4);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record record;
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kCorruption);
}

// Defense against a corrupted length field causing an unbounded/huge
// read attempt: hand-write a header whose key_length is far beyond
// kMaxKeyLength and confirm the reader rejects it immediately as
// corruption rather than trying to read gigabytes off disk.
TEST(WalCorruption, Invariant7_OversizedDeclaredLengthIsRejectedNotReadAsHugeAllocation) {
  TempFile file;
  std::string bogus_body;
  util::PutFixed32(&bogus_body, kMagic);
  bogus_body.push_back(static_cast<char>(kFormatVersion));
  bogus_body.push_back(static_cast<char>(RecordType::kPut));
  util::PutFixed64(&bogus_body, 1);
  util::PutFixed32(&bogus_body, 0xFFFFFFFFu);  // key_length: absurd
  util::PutFixed32(&bogus_body, 0u);           // value_length

  std::string encoded;
  // Checksum is deliberately not made to match — either way this must be
  // rejected, but computing a "correct" checksum for a declared 4-billion
  // byte key is not meaningful since no such payload exists on disk.
  util::PutFixed32(&encoded, 0u);
  encoded += bogus_body;
  pebbledb::testutil::WriteWholeFile(file.path(), encoded);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record record;
  ReadResult result = reader->ReadNext(&record);
  EXPECT_EQ(result, ReadResult::kCorruption);
}

TEST(WalCorruption, Invariant7_FutureFormatVersionIsReportedDistinctlyFromCorruption) {
  TempFile file;
  WriteSampleWal(file.path(), 1);

  // format_version byte is at offset 8 of the first record.
  std::string contents = pebbledb::testutil::ReadWholeFile(file.path());
  ASSERT_GE(contents.size(), kHeaderSize);
  // Recompute nothing: we intentionally do NOT fix up the checksum here,
  // because a future-version record's checksum coverage/layout is by
  // definition unknown to this (v1) reader — the point of this test is
  // that version is checked before checksum verification is even
  // attempted with v1's field layout.
  contents[8] = static_cast<char>(kFormatVersion + 1);
  pebbledb::testutil::WriteWholeFile(file.path(), contents);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record record;
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kUnsupportedVersion);
}

TEST(WalCorruption, ReaderStaysExhaustedAfterFirstCorruptionOnRepeatedCalls) {
  TempFile file;
  WriteSampleWal(file.path(), 3);
  CorruptByteAt(file.path(), 0);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record record;
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kCorruption);
  // Calling again must not crash and must return the same terminal
  // result without attempting to read further.
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kCorruption);
  EXPECT_EQ(reader->ReadNext(&record), ReadResult::kCorruption);
}

}  // namespace
}  // namespace pebbledb::wal
