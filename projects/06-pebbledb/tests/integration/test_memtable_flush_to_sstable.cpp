#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "pebbledb/memtable/memtable_list.hpp"
#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/sstable/writer.hpp"
#include "temp_file.hpp"

// End-to-end demonstration that Phase 2's "flush can snapshot immutable
// state" and Phase 3's "table written by writer reopens and queries"
// exit criteria compose correctly: an immutable MemTable's entries() feed
// directly into an sstable::Writer (the shape Phase 4's real flush
// routine will use), and the resulting on-disk table reproduces exactly
// the frozen MemTable's key/value/tombstone state after being reopened.
//
// This does not wire MemTableList into DB (that is Phase 4's "manifest +
// flush" work) -- it is a standalone proof that the two data structures
// this phase-pair delivers actually fit together the way
// docs/architecture.md's write path says they should.

namespace pebbledb {
namespace {

using pebbledb::testutil::TempFile;

// Flushes every entry in `table` (assumed already sorted -- MemTable's
// entries() guarantee) into a brand-new SSTable at `path`.
void FlushMemTableToSstable(const memtable::MemTable& table, const std::string& path) {
  std::unique_ptr<sstable::Writer> writer;
  ASSERT_TRUE(sstable::Writer::Open(path, sstable::Writer::Options{}, &writer).ok());
  for (const auto& [key, entry] : table.entries()) {
    ASSERT_TRUE(writer->Add(key, entry.value, entry.sequence, entry.type).ok()) << "key=" << key;
  }
  ASSERT_TRUE(writer->Finish().ok());
  EXPECT_EQ(writer->entries_written(), table.entry_count());
}

TEST(MemTableFlushToSstable, FrozenTableFlushesAndReopensWithIdenticalState) {
  memtable::MemTableList list;
  list.Put("apple", "red", 1);
  list.Put("banana", "yellow", 2);
  list.Delete("cherry", 3);  // tombstone, never had a live value in this table
  list.Put("date", "brown", 4);
  list.Put("date", "brown-updated", 5);  // overwritten before freeze

  std::shared_ptr<memtable::MemTable> frozen = list.Freeze();
  ASSERT_EQ(frozen->entry_count(), 4u);

  TempFile file;
  FlushMemTableToSstable(*frozen, file.path());

  std::unique_ptr<sstable::Reader> reader;
  ASSERT_TRUE(sstable::Reader::Open(file.path(), &reader).ok());
  EXPECT_EQ(reader->num_entries(), 4u);

  for (const auto& [key, memtable_entry] : frozen->entries()) {
    LookupResult result;
    std::string value;
    std::uint64_t sequence = 0;
    ASSERT_TRUE(reader->Get(key, &result, &value, &sequence).ok()) << "key=" << key;

    if (memtable_entry.type == EntryType::kTombstone) {
      EXPECT_EQ(result, LookupResult::kDeleted) << "key=" << key;
    } else {
      ASSERT_EQ(result, LookupResult::kFound) << "key=" << key;
      EXPECT_EQ(value, memtable_entry.value) << "key=" << key;
    }
    EXPECT_EQ(sequence, memtable_entry.sequence) << "key=" << key;
  }
}

// Invariant 4 (spec §1.5), carried all the way through a flush: a
// tombstone written to a MemTable must still read back as a tombstone
// (not "not found", and never a stale resurrected value) once it has
// gone through Writer -> on-disk bytes -> Reader.
TEST(MemTableFlushToSstable, Invariant4_TombstonePersistsThroughFlushAndReopen) {
  memtable::MemTableList list;
  list.Put("k1", "original", 1);
  list.Delete("k1", 2);

  std::shared_ptr<memtable::MemTable> frozen = list.Freeze();

  TempFile file;
  FlushMemTableToSstable(*frozen, file.path());

  std::unique_ptr<sstable::Reader> reader;
  ASSERT_TRUE(sstable::Reader::Open(file.path(), &reader).ok());

  LookupResult result;
  std::string value;
  ASSERT_TRUE(reader->Get("k1", &result, &value).ok());
  EXPECT_EQ(result, LookupResult::kDeleted);
}

// Randomized: build a MemTable via a stream of Put/Delete against an
// oracle map, freeze it, flush it, and confirm the reopened SSTable
// agrees with the oracle on every key -- the sstable counterpart of
// test_memtable.cpp's RandomizedOperationsMatchIndependentOracle, now
// carried one layer further through an actual on-disk round trip.
TEST(MemTableFlushToSstable, RandomizedMemTableStateSurvivesFlushAndReopen) {
  memtable::MemTableList list;
  struct OracleEntry {
    bool is_tombstone;
    std::string value;
    std::uint64_t sequence;
  };
  std::map<std::string, OracleEntry> oracle;

  std::mt19937 rng(0x5EED);
  std::uniform_int_distribution<int> op_dist(0, 1);  // put/delete only -- Get is checked below
  std::uniform_int_distribution<int> key_dist(0, 60);
  std::uniform_int_distribution<int> val_len_dist(0, 20);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  std::uint64_t sequence = 1;
  for (int i = 0; i < 2000; ++i, ++sequence) {
    std::ostringstream key_stream;
    key_stream << "key" << key_dist(rng);
    std::string key = key_stream.str();

    if (op_dist(rng) == 0) {
      std::string value;
      int len = val_len_dist(rng);
      for (int b = 0; b < len; ++b) {
        value.push_back(static_cast<char>(static_cast<unsigned char>(byte_dist(rng))));
      }
      list.Put(key, value, sequence);
      oracle[key] = OracleEntry{false, value, sequence};
    } else {
      list.Delete(key, sequence);
      oracle[key] = OracleEntry{true, "", sequence};
    }
  }

  std::shared_ptr<memtable::MemTable> frozen = list.Freeze();
  ASSERT_EQ(frozen->entry_count(), oracle.size());

  TempFile file;
  FlushMemTableToSstable(*frozen, file.path());

  std::unique_ptr<sstable::Reader> reader;
  ASSERT_TRUE(sstable::Reader::Open(file.path(), &reader).ok());
  EXPECT_EQ(reader->num_entries(), oracle.size());

  for (const auto& [key, oracle_entry] : oracle) {
    LookupResult result;
    std::string value;
    std::uint64_t seq = 0;
    ASSERT_TRUE(reader->Get(key, &result, &value, &seq).ok()) << "key=" << key;
    if (oracle_entry.is_tombstone) {
      EXPECT_EQ(result, LookupResult::kDeleted) << "key=" << key;
    } else {
      ASSERT_EQ(result, LookupResult::kFound) << "key=" << key;
      EXPECT_EQ(value, oracle_entry.value) << "key=" << key;
    }
    EXPECT_EQ(seq, oracle_entry.sequence) << "key=" << key;
  }
}

}  // namespace
}  // namespace pebbledb
