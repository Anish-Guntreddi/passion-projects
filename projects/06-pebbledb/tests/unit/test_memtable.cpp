#include "pebbledb/memtable/memtable.hpp"

#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// Phase 2 deliverable (spec roadmap): "Ordered structure, sequence/
// tombstone semantics, size threshold." These tests exercise
// memtable::MemTable in isolation -- no MemTableList merge behavior (see
// test_memtable_list.cpp for that) and no persistence.

namespace pebbledb::memtable {
namespace {

TEST(MemTable, GetOnNeverSeenKeyReturnsNotFound) {
  MemTable table;
  std::string value;
  EXPECT_EQ(table.Get("missing", &value), LookupResult::kNotFound);
}

TEST(MemTable, PutThenGetReturnsValueAndSequence) {
  MemTable table;
  table.Put("k1", "v1", 42);

  std::string value;
  std::uint64_t sequence = 0;
  ASSERT_EQ(table.Get("k1", &value, &sequence), LookupResult::kFound);
  EXPECT_EQ(value, "v1");
  EXPECT_EQ(sequence, 42u);
}

TEST(MemTable, OverwriteReplacesValueAndSequence) {
  MemTable table;
  table.Put("k1", "v1", 1);
  table.Put("k1", "v2", 2);

  std::string value;
  std::uint64_t sequence = 0;
  ASSERT_EQ(table.Get("k1", &value, &sequence), LookupResult::kFound);
  EXPECT_EQ(value, "v2");
  EXPECT_EQ(sequence, 2u);
  // Only one entry exists per key -- the earlier write is gone, not
  // retained as a second version.
  EXPECT_EQ(table.entry_count(), 1u);
}

// Invariant 4 support (spec §1.5): a delete must be visible as a distinct
// state from "never seen this key" -- see memtable.hpp, "why tombstones,
// not erasure".
TEST(MemTable, DeleteRecordsTombstoneNotErasure) {
  MemTable table;
  table.Put("k1", "v1", 1);
  ASSERT_EQ(table.entry_count(), 1u);

  table.Delete("k1", 2);

  std::string value;
  std::uint64_t sequence = 0;
  EXPECT_EQ(table.Get("k1", &value, &sequence), LookupResult::kDeleted);
  EXPECT_EQ(sequence, 2u);
  // The entry is still present (as a tombstone), not erased.
  EXPECT_EQ(table.entry_count(), 1u);
}

TEST(MemTable, DeleteOnNeverSeenKeyStillRecordsTombstone) {
  MemTable table;
  table.Delete("never-put", 1);

  std::string value;
  EXPECT_EQ(table.Get("never-put", &value), LookupResult::kDeleted);
  EXPECT_EQ(table.entry_count(), 1u);
}

TEST(MemTable, PutAfterDeleteResurrectsAsLiveValue) {
  MemTable table;
  table.Put("k1", "old", 1);
  table.Delete("k1", 2);
  table.Put("k1", "new", 3);

  std::string value;
  std::uint64_t sequence = 0;
  ASSERT_EQ(table.Get("k1", &value, &sequence), LookupResult::kFound);
  EXPECT_EQ(value, "new");
  EXPECT_EQ(sequence, 3u);
}

TEST(MemTable, DeleteAfterPutShadowsTheValue) {
  MemTable table;
  table.Put("k1", "v1", 1);
  table.Delete("k1", 2);
  table.Put("k2", "v2", 3);
  table.Delete("k2", 4);

  std::string value;
  EXPECT_EQ(table.Get("k1", &value), LookupResult::kDeleted);
  EXPECT_EQ(table.Get("k2", &value), LookupResult::kDeleted);
}

// invariant 3 support (once fed into a Phase 3 SSTable writer): entries()
// must always iterate in ascending key order regardless of insertion
// order.
TEST(MemTable, EntriesIterateInAscendingKeyOrder) {
  MemTable table;
  table.Put("banana", "v", 1);
  table.Put("apple", "v", 2);
  table.Put("cherry", "v", 3);
  table.Delete("apricot", 4);

  std::vector<std::string> keys;
  for (const auto& [key, entry] : table.entries()) {
    keys.push_back(key);
  }
  ASSERT_EQ(keys.size(), 4u);
  EXPECT_EQ(keys[0], "apple");
  EXPECT_EQ(keys[1], "apricot");
  EXPECT_EQ(keys[2], "banana");
  EXPECT_EQ(keys[3], "cherry");
}

TEST(MemTable, EmptyTableHasNoEntriesAndIsEmpty) {
  MemTable table;
  EXPECT_TRUE(table.empty());
  EXPECT_EQ(table.entry_count(), 0u);
  EXPECT_EQ(table.ApproximateMemoryUsage(), 0u);
}

TEST(MemTable, EmptyKeyAndEmptyValueAreValidAndDistinctFromMissing) {
  MemTable table;
  table.Put("", "value-for-empty-key", 1);
  std::string value;
  ASSERT_EQ(table.Get("", &value), LookupResult::kFound);
  EXPECT_EQ(value, "value-for-empty-key");

  table.Put("key-with-empty-value", "", 2);
  value = "sentinel";
  ASSERT_EQ(table.Get("key-with-empty-value", &value), LookupResult::kFound);
  EXPECT_EQ(value, "");
}

TEST(MemTable, BinarySafeKeysAndValuesRoundTripExactly) {
  MemTable table;
  std::string key("k\0ey", 4);
  std::string val;
  for (int b = 0; b < 256; ++b) {
    val.push_back(static_cast<char>(static_cast<unsigned char>(b)));
  }
  table.Put(key, val, 1);

  std::string out;
  ASSERT_EQ(table.Get(key, &out), LookupResult::kFound);
  ASSERT_EQ(out.size(), 256u);
  EXPECT_EQ(out, val);
}

// Byte/string ownership rule (slice.hpp, ADR 0002), same contract as
// DB::Put: Put()/Delete() must copy bytes rather than alias the caller's
// buffer.
TEST(MemTable, PutCopiesBytesRatherThanAliasingCallerBuffer) {
  MemTable table;
  std::string key_buf = "the-key";
  std::string val_buf = "original-value";
  table.Put(key_buf, val_buf, 1);

  key_buf.assign(key_buf.size(), 'X');
  val_buf.assign(val_buf.size(), 'X');

  std::string out;
  ASSERT_EQ(table.Get("the-key", &out), LookupResult::kFound);
  EXPECT_EQ(out, "original-value");
}

TEST(MemTable, ApproximateMemoryUsageGrowsOnInsert) {
  MemTable table;
  EXPECT_EQ(table.ApproximateMemoryUsage(), 0u);
  table.Put("k1", "v1", 1);
  const std::size_t after_one = table.ApproximateMemoryUsage();
  EXPECT_GT(after_one, 0u);
  table.Put("k2", "v2", 2);
  EXPECT_GT(table.ApproximateMemoryUsage(), after_one);
}

// Overwriting a key must adjust the running total by the *delta*, not
// double-count the old entry's bytes forever.
TEST(MemTable, ApproximateMemoryUsageAdjustsOnOverwriteRatherThanAccumulating) {
  MemTable table;
  table.Put("k1", std::string(10, 'a'), 1);
  const std::size_t after_short = table.ApproximateMemoryUsage();

  table.Put("k1", std::string(100, 'b'), 2);
  const std::size_t after_long = table.ApproximateMemoryUsage();
  EXPECT_GT(after_long, after_short);
  EXPECT_EQ(after_long - after_short, 90u);  // value grew by 90 bytes, key unchanged

  table.Put("k1", std::string(10, 'c'), 3);
  const std::size_t after_short_again = table.ApproximateMemoryUsage();
  EXPECT_EQ(after_short_again, after_short);  // back to exactly the original cost
}

TEST(MemTable, ApproximateMemoryUsageAccountsForTombstoneReplacingValue) {
  MemTable table;
  table.Put("k1", std::string(50, 'a'), 1);
  const std::size_t with_value = table.ApproximateMemoryUsage();

  table.Delete("k1", 2);
  const std::size_t with_tombstone = table.ApproximateMemoryUsage();
  // A tombstone carries no value bytes, so the cost must shrink back to
  // just the key + per-entry overhead.
  EXPECT_LT(with_tombstone, with_value);
}

TEST(MemTable, DefaultSizeThresholdIsFourMebibytes) {
  EXPECT_EQ(MemTable::kDefaultSizeThresholdBytes, 4u * 1024 * 1024);
}

// A modest randomized-operation check against an independent std::map +
// tombstone-set oracle, mirroring the style and intent of
// tests/unit/test_db_map_reference.cpp's RandomizedOperationsMatchIndependentOracle.
TEST(MemTable, RandomizedOperationsMatchIndependentOracle) {
  MemTable table;
  // oracle: key -> {is_tombstone, value, sequence}
  struct OracleEntry {
    bool is_tombstone;
    std::string value;
    std::uint64_t sequence;
  };
  std::map<std::string, OracleEntry> oracle;

  std::mt19937 rng(0xBADA55);
  std::uniform_int_distribution<int> op_dist(0, 2);
  std::uniform_int_distribution<int> key_dist(0, 24);
  std::uniform_int_distribution<int> val_len_dist(0, 12);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  std::uint64_t sequence = 1;
  for (int i = 0; i < 5000; ++i, ++sequence) {
    std::ostringstream key_stream;
    key_stream << "key" << key_dist(rng);
    std::string key = key_stream.str();

    int op = op_dist(rng);
    if (op == 0) {  // Put
      std::string value;
      int len = val_len_dist(rng);
      for (int b = 0; b < len; ++b) {
        value.push_back(static_cast<char>(static_cast<unsigned char>(byte_dist(rng))));
      }
      table.Put(key, value, sequence);
      oracle[key] = OracleEntry{false, value, sequence};
    } else if (op == 1) {  // Get
      std::string actual;
      std::uint64_t actual_seq = 0;
      LookupResult result = table.Get(key, &actual, &actual_seq);
      auto it = oracle.find(key);
      if (it == oracle.end()) {
        EXPECT_EQ(result, LookupResult::kNotFound) << "key=" << key;
      } else if (it->second.is_tombstone) {
        EXPECT_EQ(result, LookupResult::kDeleted) << "key=" << key;
        EXPECT_EQ(actual_seq, it->second.sequence) << "key=" << key;
      } else {
        ASSERT_EQ(result, LookupResult::kFound) << "key=" << key;
        EXPECT_EQ(actual, it->second.value) << "key=" << key;
        EXPECT_EQ(actual_seq, it->second.sequence) << "key=" << key;
      }
    } else {  // Delete
      table.Delete(key, sequence);
      oracle[key] = OracleEntry{true, "", sequence};
    }
  }

  // Final full sweep, and confirm entries() iterates in ascending order
  // while we're at it.
  std::string previous_key;
  bool has_previous = false;
  for (const auto& [key, entry] : table.entries()) {
    if (has_previous) {
      EXPECT_LT(previous_key, key);
    }
    previous_key = key;
    has_previous = true;

    auto it = oracle.find(key);
    ASSERT_NE(it, oracle.end()) << "key=" << key;
    if (it->second.is_tombstone) {
      EXPECT_EQ(entry.type, EntryType::kTombstone) << "key=" << key;
    } else {
      EXPECT_EQ(entry.type, EntryType::kValue) << "key=" << key;
      EXPECT_EQ(entry.value, it->second.value) << "key=" << key;
    }
    EXPECT_EQ(entry.sequence, it->second.sequence) << "key=" << key;
  }
  EXPECT_EQ(table.entry_count(), oracle.size());
}

}  // namespace
}  // namespace pebbledb::memtable
