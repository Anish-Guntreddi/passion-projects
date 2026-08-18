#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/sstable/writer.hpp"
#include "temp_file.hpp"

// Phase 3 exit criterion (spec roadmap): "Table written by writer reopens
// and queries after process restart." These tests write real SSTable
// files through sstable::Writer, destroy the Writer object (the same
// "simulated crash/restart" pattern tests/recovery uses for the WAL --
// nothing survives in-process except what actually landed on disk), then
// open a brand-new sstable::Reader on that same path and confirm every
// query resolves correctly. Real-file corruption is covered separately in
// tests/corruption/test_sstable_corruption.cpp.

namespace pebbledb::sstable {
namespace {

using pebbledb::testutil::TempFile;

struct SampleEntry {
  std::string key;
  std::uint64_t sequence;
  EntryType type;
  std::string value;  // empty/meaningless for kTombstone
};

// Builds `count` sorted, distinct keys ("key0000".."keyNNNN"), alternating
// live values and tombstones every 5th key.
std::vector<SampleEntry> BuildSampleEntries(int count) {
  std::vector<SampleEntry> entries;
  entries.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "key%05d", i);
    SampleEntry e;
    e.key = buf;
    e.sequence = static_cast<std::uint64_t>(i + 1);
    if (i % 5 == 4) {
      e.type = EntryType::kTombstone;
    } else {
      e.type = EntryType::kValue;
      e.value = "value-" + std::to_string(i) + "-" + std::string(static_cast<std::size_t>(i % 37), 'x');
    }
    entries.push_back(std::move(e));
  }
  return entries;
}

void WriteSampleTable(const std::string& path, const std::vector<SampleEntry>& entries,
                      const Writer::Options& options = Writer::Options{}) {
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(path, options, &writer).ok());
  for (const SampleEntry& e : entries) {
    ASSERT_TRUE(writer->Add(e.key, e.value, e.sequence, e.type).ok()) << "key=" << e.key;
  }
  ASSERT_TRUE(writer->Finish().ok());
  EXPECT_EQ(writer->entries_written(), entries.size());
}

TEST(SstableWriterReader, WriteThenReopenAndGetEveryKey) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(50);
  WriteSampleTable(file.path(), entries);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  EXPECT_EQ(reader->num_entries(), entries.size());

  for (const SampleEntry& e : entries) {
    LookupResult result;
    std::string value;
    std::uint64_t sequence = 0;
    ASSERT_TRUE(reader->Get(e.key, &result, &value, &sequence).ok()) << "key=" << e.key;
    if (e.type == EntryType::kTombstone) {
      EXPECT_EQ(result, LookupResult::kDeleted) << "key=" << e.key;
    } else {
      ASSERT_EQ(result, LookupResult::kFound) << "key=" << e.key;
      EXPECT_EQ(value, e.value) << "key=" << e.key;
    }
    EXPECT_EQ(sequence, e.sequence) << "key=" << e.key;
  }
}

TEST(SstableWriterReader, GetOnMissingKeysReturnsNotFound) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(20);
  WriteSampleTable(file.path(), entries);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());

  LookupResult result;
  std::string value;
  // Before the first key.
  ASSERT_TRUE(reader->Get("aaa-before-everything", &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kNotFound);
  // After the last key.
  ASSERT_TRUE(reader->Get("zzz-after-everything", &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kNotFound);
  // Between two present keys, but not itself present.
  ASSERT_TRUE(reader->Get("key00001x", &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kNotFound);
}

// Forces many small blocks (tiny target size) so this exercises the
// sparse index / multi-block binary search path, not just a
// single-block table.
TEST(SstableWriterReader, KeysSpanningManyBlocksAreAllFoundCorrectly) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(200);
  Writer::Options options;
  options.target_block_size_bytes = 64;  // forces many blocks
  WriteSampleTable(file.path(), entries, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  // 200 small entries at a 64-byte target must produce more than one block.
  EXPECT_GT(reader->index_entry_count(), 1u);

  for (const SampleEntry& e : entries) {
    LookupResult result;
    std::string value;
    ASSERT_TRUE(reader->Get(e.key, &result, &value).ok()) << "key=" << e.key;
    if (e.type == EntryType::kTombstone) {
      EXPECT_EQ(result, LookupResult::kDeleted) << "key=" << e.key;
    } else {
      ASSERT_EQ(result, LookupResult::kFound) << "key=" << e.key;
      EXPECT_EQ(value, e.value) << "key=" << e.key;
    }
  }
}

TEST(SstableWriterReader, ForEachEntryVisitsAllEntriesInAscendingOrder) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(100);
  Writer::Options options;
  options.target_block_size_bytes = 200;
  WriteSampleTable(file.path(), entries, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());

  std::vector<std::string> visited_keys;
  Status s = reader->ForEachEntry([&](const Entry& entry) { visited_keys.push_back(entry.key); });
  ASSERT_TRUE(s.ok());

  ASSERT_EQ(visited_keys.size(), entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    EXPECT_EQ(visited_keys[i], entries[i].key) << "index=" << i;
  }
}

// Invariant 3 (spec §1.5): "Keys inside a sorted table obey comparator
// order." Writer must reject an attempt to violate that, rather than
// silently producing an unsorted (and therefore un-binary-searchable)
// file.
TEST(SstableWriterReader, Invariant3_WriterRejectsOutOfOrderKey) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());

  ASSERT_TRUE(writer->Add("banana", "v", 1, EntryType::kValue).ok());
  Status s = writer->Add("apple", "v", 2, EntryType::kValue);  // "apple" < "banana"
  EXPECT_TRUE(s.IsInvalidArgument());
}

TEST(SstableWriterReader, Invariant3_WriterRejectsDuplicateKey) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());

  ASSERT_TRUE(writer->Add("k1", "v1", 1, EntryType::kValue).ok());
  Status s = writer->Add("k1", "v2", 2, EntryType::kValue);  // duplicate, not strictly greater
  EXPECT_TRUE(s.IsInvalidArgument());
}

// Invariant 2 (spec §1.5): "SSTable contents are immutable after
// successful creation." Add()/Finish() must refuse to touch the file
// again once Finish() has completed.
TEST(SstableWriterReader, Invariant2_WriterRejectsAddAfterFinish) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());
  ASSERT_TRUE(writer->Add("k1", "v1", 1, EntryType::kValue).ok());
  ASSERT_TRUE(writer->Finish().ok());

  Status s = writer->Add("k2", "v2", 2, EntryType::kValue);
  EXPECT_TRUE(s.IsInvalidArgument());
}

TEST(SstableWriterReader, Invariant2_WriterRejectsFinishCalledTwice) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());
  ASSERT_TRUE(writer->Add("k1", "v1", 1, EntryType::kValue).ok());
  ASSERT_TRUE(writer->Finish().ok());

  Status s = writer->Finish();
  EXPECT_TRUE(s.IsInvalidArgument());
}

// Regression coverage (ADR 0011): Writer::Open() must fail outright
// against a path that already has a complete, Finish()ed table at it,
// rather than silently opening it with util::PosixFile::OpenAppend
// (O_APPEND) and going on to record new BlockHandles starting from
// file_offset_ == 0 while the new bytes actually land after the old
// table's content on disk. Reproduces the exact scenario a prior review
// found: a second Writer pointed at the first table's path.
TEST(SstableWriterReader, Invariant2_WriterOpenOnExistingFileFailsInsteadOfCorruptingIt) {
  TempFile file;

  // First table: a single real entry, fully finished and durable.
  std::unique_ptr<Writer> first_writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &first_writer).ok());
  ASSERT_TRUE(first_writer->Add("a", "first-table-value", 1, EntryType::kValue).ok());
  ASSERT_TRUE(first_writer->Finish().ok());
  const std::int64_t size_after_first_table = pebbledb::testutil::FileSize(file.path());
  ASSERT_GT(size_after_first_table, 0);

  // A second Writer::Open() against the exact same (already-existing,
  // non-empty) path must fail outright -- not silently append a second
  // table's worth of bytes onto the first.
  std::unique_ptr<Writer> second_writer;
  Status open_status = Writer::Open(file.path(), Writer::Options{}, &second_writer);
  EXPECT_FALSE(open_status.ok());
  EXPECT_EQ(second_writer, nullptr);

  // The first table's on-disk bytes must be completely untouched: same
  // size, and still correctly readable end-to-end.
  EXPECT_EQ(pebbledb::testutil::FileSize(file.path()), size_after_first_table);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  LookupResult result;
  std::string value;
  ASSERT_TRUE(reader->Get("a", &result, &value).ok());
  ASSERT_EQ(result, LookupResult::kFound);
  EXPECT_EQ(value, "first-table-value");
}

// An SSTable with zero entries is a valid, if degenerate, file: no data
// blocks, an empty index block, and a footer reporting num_entries == 0.
TEST(SstableWriterReader, EmptyTableIsValidAndReopens) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());
  ASSERT_TRUE(writer->Finish().ok());
  EXPECT_EQ(writer->entries_written(), 0u);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  EXPECT_EQ(reader->num_entries(), 0u);
  EXPECT_EQ(reader->index_entry_count(), 0u);

  LookupResult result;
  std::string value;
  ASSERT_TRUE(reader->Get("anything", &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kNotFound);
}

// Phase 3-era behavior (ADR 0009): the filter block was *always* absent,
// unconditionally. Phase 5 (ADR 0014) makes filter generation the
// default (Writer::Options::filter_bits_per_key defaults to 10) -- see
// FilterHandleIsPresentByDefault below for that case. The filter block
// is still absent when explicitly disabled (filter_bits_per_key == 0),
// which is what this test now covers -- the "reserved, size == 0 means
// absent" footer contract itself is unchanged either way.
TEST(SstableWriterReader, FilterHandleIsAbsentWhenFilterDisabled) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(10);
  Writer::Options options;
  options.filter_bits_per_key = 0;
  WriteSampleTable(file.path(), entries, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  EXPECT_EQ(reader->footer().filter_handle.size, 0u);
  EXPECT_FALSE(reader->has_filter());
}

// Phase 5 (spec FR3, roadmap Phase 5; ADR 0014): Writer::Options's
// default (filter_bits_per_key == 10) produces a real, non-empty filter
// block that Reader::Open() loads.
TEST(SstableWriterReader, FilterHandleIsPresentByDefault) {
  TempFile file;
  std::vector<SampleEntry> entries = BuildSampleEntries(10);
  WriteSampleTable(file.path(), entries);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  EXPECT_GT(reader->footer().filter_handle.size, 0u);
  EXPECT_TRUE(reader->has_filter());
}

TEST(SstableWriterReader, LargeKeyAndValueRoundTrip) {
  TempFile file;
  std::unique_ptr<Writer> writer;
  ASSERT_TRUE(Writer::Open(file.path(), Writer::Options{}, &writer).ok());

  std::string key(5000, 'k');
  std::string value(70000, 'v');
  ASSERT_TRUE(writer->Add(key, value, 1, EntryType::kValue).ok());
  ASSERT_TRUE(writer->Finish().ok());

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  LookupResult result;
  std::string out;
  ASSERT_TRUE(reader->Get(key, &result, &out).ok());
  ASSERT_EQ(result, LookupResult::kFound);
  EXPECT_EQ(out, value);
}

TEST(SstableWriterReader, ReaderOpenOnNonexistentPathFails) {
  std::string path = pebbledb::testutil::UniqueNonExistentPath();
  std::unique_ptr<Reader> reader;
  Status s = Reader::Open(path, &reader);
  EXPECT_FALSE(s.ok());
}

TEST(SstableWriterReader, WriterOpenOnBrandNewPathSucceedsAndReopens) {
  std::string path = pebbledb::testutil::UniqueNonExistentPath();
  ASSERT_EQ(pebbledb::testutil::FileSize(path), -1);  // must not exist yet

  {
    std::unique_ptr<Writer> writer;
    ASSERT_TRUE(Writer::Open(path, Writer::Options{}, &writer).ok());
    ASSERT_TRUE(writer->Add("k", "v", 1, EntryType::kValue).ok());
    ASSERT_TRUE(writer->Finish().ok());
  }

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(path, &reader).ok());
  LookupResult result;
  std::string value;
  ASSERT_TRUE(reader->Get("k", &result, &value).ok());
  ASSERT_EQ(result, LookupResult::kFound);
  EXPECT_EQ(value, "v");

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

}  // namespace
}  // namespace pebbledb::sstable
