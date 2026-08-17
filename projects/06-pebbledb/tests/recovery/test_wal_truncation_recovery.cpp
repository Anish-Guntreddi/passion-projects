#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "pebbledb/wal/reader.hpp"
#include "pebbledb/wal/writer.hpp"
#include "temp_file.hpp"

// Phase 1 exit criterion (spec roadmap): "Write sequence replays
// deterministically after simulated crash." Spec §1.8 "Recovery" test
// strategy: "write records -> truncate at every possible record boundary
// -> reopen; acknowledged records survive per sync semantics."
//
// "Simulated crash", precisely, for what a hosted unit test can actually
// verify (see docs/durability.md for the full discussion): this suite
// (a) writes a sequence of records, fsyncing per a chosen SyncPolicy, (b)
// destroys the Writer with no special shutdown step (the closest a unit
// test gets to an abrupt process exit — the Writer keeps no additional
// in-process buffer beyond what SyncPolicy has already decided to flush),
// then (c) opens a brand-new Reader on the same path (modeling the
// recovering process) and replays. It does NOT and cannot simulate real
// kernel/power-loss data loss (that would require dropping the OS page
// cache or cutting power to real hardware, unavailable in CI); what it
// verifies instead is that replay is byte-for-byte deterministic given
// whatever is actually on disk, and that truncation at any byte offset
// (not just at record boundaries) is handled without ever fabricating a
// record that wasn't fully and validly written.

namespace pebbledb::wal {
namespace {

using pebbledb::testutil::FileSize;
using pebbledb::testutil::TempFile;

std::vector<Record> BuildSampleRecords() {
  std::vector<Record> records;
  for (int i = 0; i < 20; ++i) {
    Record r;
    r.sequence = static_cast<std::uint64_t>(i + 1);
    if (i % 5 == 4) {
      r.type = RecordType::kDelete;
      r.key = "key" + std::to_string(i);
    } else {
      r.type = RecordType::kPut;
      r.key = "key" + std::to_string(i);
      r.value = "value-" + std::to_string(i) + "-" + std::string(static_cast<std::size_t>(i), 'x');
    }
    records.push_back(r);
  }
  return records;
}

bool RecordsEqual(const Record& a, const Record& b) {
  return a.sequence == b.sequence && a.type == b.type && a.key == b.key && a.value == b.value;
}

// ---------------------------------------------------------------------
// Invariant 1 (spec §1.5): "An acknowledged durable write is recoverable
// after process crash per the documented sync policy." With
// SyncPolicy::kEveryWrite, AddRecord does not return OK until fsync
// succeeded, so every record in BuildSampleRecords() is "acknowledged".
// ---------------------------------------------------------------------
TEST(WalTruncationRecovery, Invariant1_AllAcknowledgedRecordsReplayAfterSimulatedCrash) {
  TempFile file;
  std::vector<Record> records = BuildSampleRecords();

  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());
    for (const Record& r : records) {
      ASSERT_TRUE(writer->AddRecord(r).ok());
    }
    // `writer` is destroyed here with no explicit "close/finalize" call —
    // the simulated crash.
  }

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result =
      reader->Replay([&](const Record& r) { replayed.push_back(r); });

  EXPECT_EQ(result.terminal, ReadResult::kEndOfLog);
  ASSERT_EQ(replayed.size(), records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    EXPECT_TRUE(RecordsEqual(replayed[i], records[i])) << "record index " << i;
  }
}

TEST(WalTruncationRecovery, ReplayIsDeterministicAcrossIndependentReaderInstances) {
  TempFile file;
  std::vector<Record> records = BuildSampleRecords();

  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());
  for (const Record& r : records) {
    ASSERT_TRUE(writer->AddRecord(r).ok());
  }

  auto replay_all = [&]() {
    std::unique_ptr<Reader> reader;
    EXPECT_TRUE(Reader::Open(file.path(), &reader).ok());
    std::vector<Record> replayed;
    Reader::ReplayResult result =
        reader->Replay([&](const Record& r) { replayed.push_back(r); });
    EXPECT_EQ(result.terminal, ReadResult::kEndOfLog);
    return replayed;
  };

  std::vector<Record> first = replay_all();
  std::vector<Record> second = replay_all();

  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_TRUE(RecordsEqual(first[i], second[i])) << "record index " << i;
  }
}

// ---------------------------------------------------------------------
// The core "truncate at every possible record boundary" test (spec
// §1.8). Exhaustive over every single byte offset from 0 to the full
// file length, not just the boundaries between whole records: this
// covers both clean boundaries (kEndOfLog) and every torn-mid-record
// case (kTruncated) a real unclean shutdown could produce.
// ---------------------------------------------------------------------
TEST(WalTruncationRecovery, ReplayIsCorrectAtEveryPossibleTruncationOffset) {
  TempFile source_file;
  std::vector<Record> records = BuildSampleRecords();

  // Write once, fsyncing every record, and record the cumulative file
  // size after each record so we know exactly which byte offsets are
  // "clean" record boundaries vs. mid-record.
  std::vector<std::int64_t> cumulative_offsets;
  cumulative_offsets.push_back(0);
  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(source_file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());
    for (const Record& r : records) {
      ASSERT_TRUE(writer->AddRecord(r).ok());
      cumulative_offsets.push_back(FileSize(source_file.path()));
    }
  }

  const std::int64_t full_length = cumulative_offsets.back();
  ASSERT_EQ(full_length, FileSize(source_file.path()));
  const std::string full_contents = pebbledb::testutil::ReadWholeFile(source_file.path());
  ASSERT_EQ(static_cast<std::int64_t>(full_contents.size()), full_length);

  for (std::int64_t truncate_len = 0; truncate_len <= full_length; ++truncate_len) {
    TempFile truncated_file;
    pebbledb::testutil::WriteWholeFile(truncated_file.path(),
                                        full_contents.substr(0, static_cast<std::size_t>(truncate_len)));

    // How many complete records fit within the first `truncate_len` bytes?
    std::size_t expected_count = 0;
    while (expected_count < records.size() &&
           cumulative_offsets[expected_count + 1] <= truncate_len) {
      ++expected_count;
    }
    const bool is_clean_boundary = (cumulative_offsets[expected_count] == truncate_len);

    std::unique_ptr<Reader> reader;
    ASSERT_TRUE(Reader::Open(truncated_file.path(), &reader).ok());
    std::vector<Record> replayed;
    Reader::ReplayResult result =
        reader->Replay([&](const Record& r) { replayed.push_back(r); });

    ASSERT_EQ(replayed.size(), expected_count) << "truncate_len=" << truncate_len;
    for (std::size_t i = 0; i < expected_count; ++i) {
      EXPECT_TRUE(RecordsEqual(replayed[i], records[i]))
          << "truncate_len=" << truncate_len << " record index " << i;
    }

    // Pure truncation (no byte modification) must never surface as
    // kCorruption: an incomplete record is kTruncated (or kEndOfLog if
    // truncate_len happens to land exactly on a record boundary), never
    // a checksum failure, since nothing that *was* written was altered.
    if (is_clean_boundary) {
      EXPECT_EQ(result.terminal, ReadResult::kEndOfLog) << "truncate_len=" << truncate_len;
    } else {
      EXPECT_EQ(result.terminal, ReadResult::kTruncated) << "truncate_len=" << truncate_len;
    }
  }
}

// ---------------------------------------------------------------------
// SyncPolicy cadence: verifies *when* each policy invokes fsync, via the
// Writer's own call counters (sync_count/records_since_sync). This is the
// deterministic, testable half of "sync policy controls durability" —
// see docs/durability.md for what remains untestable (actual power-loss
// survival) and why.
// ---------------------------------------------------------------------
TEST(WalTruncationRecovery, SyncPolicyEveryWriteSyncsAfterEachRecord) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());

  for (int i = 0; i < 5; ++i) {
    Record r;
    r.sequence = static_cast<std::uint64_t>(i);
    r.type = RecordType::kPut;
    r.key = "k";
    r.value = "v";
    ASSERT_TRUE(writer->AddRecord(r).ok());
    EXPECT_EQ(writer->sync_count(), static_cast<std::uint64_t>(i + 1));
    EXPECT_EQ(writer->records_since_sync(), 0u);
  }
}

TEST(WalTruncationRecovery, SyncPolicyBatchedSyncsEveryBatchSizeRecords) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  constexpr std::size_t kBatchSize = 5;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kBatched, kBatchSize, &writer).ok());

  for (int i = 1; i <= 12; ++i) {
    Record r;
    r.sequence = static_cast<std::uint64_t>(i);
    r.type = RecordType::kPut;
    r.key = "k";
    r.value = "v";
    ASSERT_TRUE(writer->AddRecord(r).ok());
  }
  // 12 records at batch size 5: syncs after record 5 and record 10 -> 2
  // syncs so far, 2 records pending (11, 12).
  EXPECT_EQ(writer->records_written(), 12u);
  EXPECT_EQ(writer->sync_count(), 2u);
  EXPECT_EQ(writer->records_since_sync(), 2u);

  ASSERT_TRUE(writer->Sync().ok());
  EXPECT_EQ(writer->sync_count(), 3u);
  EXPECT_EQ(writer->records_since_sync(), 0u);
}

TEST(WalTruncationRecovery, SyncPolicyExplicitNeverSyncsAutomatically) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kExplicit, 1, &writer).ok());

  for (int i = 0; i < 10; ++i) {
    Record r;
    r.sequence = static_cast<std::uint64_t>(i);
    r.type = RecordType::kPut;
    r.key = "k";
    r.value = "v";
    ASSERT_TRUE(writer->AddRecord(r).ok());
  }
  EXPECT_EQ(writer->sync_count(), 0u);
  EXPECT_EQ(writer->records_since_sync(), 10u);

  ASSERT_TRUE(writer->Sync().ok());
  EXPECT_EQ(writer->sync_count(), 1u);
  EXPECT_EQ(writer->records_since_sync(), 0u);
}

TEST(WalTruncationRecovery, OpenRejectsZeroBatchSizeForBatchedPolicy) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  Status s = Writer::Open(file.path(), SyncPolicy::kBatched, 0, &writer);
  EXPECT_TRUE(s.IsInvalidArgument());
}

// ---------------------------------------------------------------------
// Regression coverage: AddRecord() must reject an oversized key/value
// before writing or acknowledging anything, rather than silently
// encoding a record that wal::DecodeHeader() can never successfully
// decode on replay (kBadLength -> Reader reports kCorruption, which
// halts recovery at that record). See docs/durability.md and ADR 0006.
// ---------------------------------------------------------------------
TEST(WalTruncationRecovery, AddRecordRejectsOversizedKeyWithoutWritingAnything) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());

  Record r;
  r.sequence = 1;
  r.type = RecordType::kPut;
  r.key = std::string(kMaxKeyLength + 1, 'k');
  r.value = "v";

  Status s = writer->AddRecord(r);
  EXPECT_TRUE(s.IsInvalidArgument());
  EXPECT_EQ(writer->records_written(), 0u);
  EXPECT_EQ(FileSize(file.path()), 0);
}

TEST(WalTruncationRecovery, AddRecordRejectsOversizedValueWithoutWritingAnything) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());

  Record r;
  r.sequence = 1;
  r.type = RecordType::kPut;
  r.key = "k";
  r.value = std::string(kMaxValueLength + 1, 'v');

  Status s = writer->AddRecord(r);
  EXPECT_TRUE(s.IsInvalidArgument());
  EXPECT_EQ(writer->records_written(), 0u);
  EXPECT_EQ(FileSize(file.path()), 0);
}

// A kDelete record's value is never persisted (record.hpp), so an
// oversized value on a Delete must not be rejected -- it is simply
// ignored, exactly as EncodeRecord() already ignores it.
TEST(WalTruncationRecovery, AddRecordAcceptsDeleteWithOversizedIgnoredValue) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());

  Record r;
  r.sequence = 1;
  r.type = RecordType::kDelete;
  r.key = "k";
  r.value = std::string(kMaxValueLength + 1, 'v');  // must be ignored, not rejected

  EXPECT_TRUE(writer->AddRecord(r).ok());
  EXPECT_EQ(writer->records_written(), 1u);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  Record replayed;
  ASSERT_EQ(reader->ReadNext(&replayed), ReadResult::kOk);
  EXPECT_EQ(replayed.type, RecordType::kDelete);
  EXPECT_EQ(replayed.key, "k");
  EXPECT_EQ(replayed.value, "");
}

// End-to-end confirmation that a rejected oversized record does not
// disturb records written before or after the rejected call.
TEST(WalTruncationRecovery, RejectedOversizedRecordDoesNotDisturbNeighboringRecords) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());

  Record first;
  first.sequence = 1;
  first.type = RecordType::kPut;
  first.key = "before";
  first.value = "v1";
  ASSERT_TRUE(writer->AddRecord(first).ok());

  Record oversized;
  oversized.sequence = 2;
  oversized.type = RecordType::kPut;
  oversized.key = std::string(kMaxKeyLength + 1, 'k');
  oversized.value = "v";
  EXPECT_TRUE(writer->AddRecord(oversized).IsInvalidArgument());

  Record last;
  last.sequence = 3;
  last.type = RecordType::kPut;
  last.key = "after";
  last.value = "v3";
  ASSERT_TRUE(writer->AddRecord(last).ok());

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result = reader->Replay([&](const Record& r) { replayed.push_back(r); });

  EXPECT_EQ(result.terminal, ReadResult::kEndOfLog);
  ASSERT_EQ(replayed.size(), 2u);
  EXPECT_TRUE(RecordsEqual(replayed[0], first));
  EXPECT_TRUE(RecordsEqual(replayed[1], last));
}

// ---------------------------------------------------------------------
// Regression coverage: Writer::Open (via util::PosixFile::OpenAppend)
// must fsync the containing directory when it creates a brand-new WAL
// file, so the directory entry survives a crash by the time any record
// written to that file is acknowledged. TempFile pre-creates its path via
// mkstemp, so every other test in this file exercises the "path already
// existed" branch only -- these tests specifically use
// UniqueNonExistentPath() to exercise the "Open() itself creates the
// file" branch instead. See docs/durability.md and
// util/posix_file.cpp.
// ---------------------------------------------------------------------
TEST(WalTruncationRecovery, OpenOnBrandNewPathSucceedsAndReplaysCorrectly) {
  std::string path = pebbledb::testutil::UniqueNonExistentPath();
  ASSERT_EQ(FileSize(path), -1);  // must not exist yet

  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(path, SyncPolicy::kEveryWrite, 1, &writer).ok());

    Record r;
    r.sequence = 1;
    r.type = RecordType::kPut;
    r.key = "k";
    r.value = "v";
    ASSERT_TRUE(writer->AddRecord(r).ok());
  }

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(path, &reader).ok());
  Record replayed;
  ASSERT_EQ(reader->ReadNext(&replayed), ReadResult::kOk);
  EXPECT_EQ(replayed.key, "k");
  EXPECT_EQ(replayed.value, "v");

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

// Reopening an already-existing WAL path (the normal "resume appending
// after restart" case) must not error even though it takes the "path
// already existed" branch that skips the directory fsync (no new
// directory entry was created, so there is nothing new to sync).
TEST(WalTruncationRecovery, ReopeningAnExistingWalPathForAppendSucceeds) {
  TempFile file;
  std::vector<Record> records = BuildSampleRecords();

  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());
    ASSERT_TRUE(writer->AddRecord(records[0]).ok());
  }
  {
    // Reopen the same (now-existing, non-empty) path and append more.
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kEveryWrite, 1, &writer).ok());
    ASSERT_TRUE(writer->AddRecord(records[1]).ok());
  }

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result = reader->Replay([&](const Record& r) { replayed.push_back(r); });
  EXPECT_EQ(result.terminal, ReadResult::kEndOfLog);
  ASSERT_EQ(replayed.size(), 2u);
  EXPECT_TRUE(RecordsEqual(replayed[0], records[0]));
  EXPECT_TRUE(RecordsEqual(replayed[1], records[1]));
}

// Bytes written under kExplicit (without an intervening Sync()) are still
// visible to a subsequent Reader in this environment, since we're not
// simulating real power loss — only real process-level abandonment of
// the Writer object. This documents the boundary between what the test
// suite verifies (replay reflects on-disk bytes deterministically) and
// what it does not claim (survival of literal power loss) — see
// docs/durability.md.
TEST(WalTruncationRecovery, UnsyncedExplicitWritesAreStillVisibleToASubsequentReader) {
  TempFile file;
  std::vector<Record> records = BuildSampleRecords();
  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(file.path(), SyncPolicy::kExplicit, 1, &writer).ok());
    for (const Record& r : records) {
      ASSERT_TRUE(writer->AddRecord(r).ok());
    }
    // Deliberately no Sync() call before the writer is destroyed.
  }

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  std::vector<Record> replayed;
  Reader::ReplayResult result =
      reader->Replay([&](const Record& r) { replayed.push_back(r); });
  EXPECT_EQ(result.terminal, ReadResult::kEndOfLog);
  EXPECT_EQ(replayed.size(), records.size());
}

}  // namespace
}  // namespace pebbledb::wal
