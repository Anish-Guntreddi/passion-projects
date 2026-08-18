#include "pebbledb/db.hpp"

#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

// Phase 0 exit criterion (spec roadmap): "put/get/delete semantics tested
// without persistence." These tests exercise DB's in-memory-only mode
// (the default constructor, `DB db;`) directly — no WAL, no files, no
// restart. As of Phase 4 this mode is backed by memtable::MemTableList
// rather than a raw std::map (see ADR 0013), but its contract is
// unchanged: no persistence, no path argument, fully supported
// alongside (not superseded by) the disk-backed DB::Open() mode.
// Persisted-state tests (WAL-level) live in tests/recovery and
// tests/corruption; a real, disk-backed DB surviving a process restart
// is tests/integration/test_db_persistence.cpp (Phase 4).

namespace pebbledb {
namespace {

TEST(Db, GetOnEmptyDbIsNotFound) {
  DB db;
  std::string value;
  Status s = db.Get("missing", &value);
  EXPECT_TRUE(s.IsNotFound());
}

TEST(Db, PutThenGetReturnsSameValue) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  std::string value;
  ASSERT_TRUE(db.Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
}

TEST(Db, PutOverwritesPreviousValueForSameKey) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  ASSERT_TRUE(db.Put("k1", "v2").ok());
  std::string value;
  ASSERT_TRUE(db.Get("k1", &value).ok());
  EXPECT_EQ(value, "v2");
}

TEST(Db, DeleteRemovesKey) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  ASSERT_TRUE(db.Delete("k1").ok());
  std::string value;
  Status s = db.Get("k1", &value);
  EXPECT_TRUE(s.IsNotFound());
}

// Invariant 4 (spec §1.5): "Tombstone semantics prevent stale older
// values from resurfacing." At Phase 0 there is only one storage layer
// (the map itself), so this reduces to: delete-then-put must show the
// new value, never the pre-delete one, and must not resurrect a deleted
// value on repeated reads.
TEST(Db, Invariant4_PutDeletePutDoesNotResurfaceStaleValue) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "old").ok());
  ASSERT_TRUE(db.Delete("k1").ok());
  ASSERT_TRUE(db.Put("k1", "new").ok());

  std::string value;
  ASSERT_TRUE(db.Get("k1", &value).ok());
  EXPECT_EQ(value, "new");

  // Read again: still "new", not a stale resurfacing of "old".
  value.clear();
  ASSERT_TRUE(db.Get("k1", &value).ok());
  EXPECT_EQ(value, "new");
}

TEST(Db, DeleteOnAbsentKeyIsIdempotentOk) {
  DB db;
  Status s = db.Delete("never-existed");
  EXPECT_TRUE(s.ok());

  // Deleting an already-deleted key is also OK.
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  ASSERT_TRUE(db.Delete("k1").ok());
  Status s2 = db.Delete("k1");
  EXPECT_TRUE(s2.ok());
}

TEST(Db, EmptyKeyAndEmptyValueAreValidAndDistinctFromMissing) {
  DB db;
  ASSERT_TRUE(db.Put("", "value-for-empty-key").ok());
  std::string value;
  ASSERT_TRUE(db.Get("", &value).ok());
  EXPECT_EQ(value, "value-for-empty-key");

  ASSERT_TRUE(db.Put("key-with-empty-value", "").ok());
  value = "sentinel-should-be-cleared";
  ASSERT_TRUE(db.Get("key-with-empty-value", &value).ok());
  EXPECT_EQ(value, "");  // present with an empty value, not "not found"
}

TEST(Db, BinarySafeKeysAndValuesRoundTripExactly) {
  DB db;
  std::string key("k\0ey", 4);
  std::string val;
  for (int b = 0; b < 256; ++b) {
    val.push_back(static_cast<char>(static_cast<unsigned char>(b)));
  }
  ASSERT_TRUE(db.Put(key, val).ok());

  std::string out;
  ASSERT_TRUE(db.Get(key, &out).ok());
  ASSERT_EQ(out.size(), 256u);
  EXPECT_EQ(out, val);
}

// Byte/string ownership rule (slice.hpp, ADR 0002): Put() must copy the
// bytes it's given into DB-owned storage. If it instead aliased the
// caller's buffer, mutating that buffer after Put() would corrupt the
// stored value.
TEST(Db, PutCopiesBytesRatherThanAliasingCallerBuffer) {
  DB db;
  std::string key_buf = "the-key";
  std::string val_buf = "original-value";

  ASSERT_TRUE(db.Put(key_buf, val_buf).ok());

  // Mutate the caller's buffers after Put() returns.
  key_buf.assign(key_buf.size(), 'X');
  val_buf.assign(val_buf.size(), 'X');

  std::string out;
  ASSERT_TRUE(db.Get("the-key", &out).ok());
  EXPECT_EQ(out, "original-value");
}

TEST(Db, GetCopiesIntoOutputRatherThanAliasingInternalStorage) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  std::string out;
  ASSERT_TRUE(db.Get("k1", &out).ok());
  out.append("-mutated-by-caller");

  std::string out2;
  ASSERT_TRUE(db.Get("k1", &out2).ok());
  EXPECT_EQ(out2, "v1");  // unaffected by mutating the first Get's output
}

TEST(Db, FlushAndCompactAreNoOpsThatReturnOkInPhase0) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  EXPECT_TRUE(db.Flush().ok());
  EXPECT_TRUE(db.Compact().ok());

  // Data is unaffected — there was nothing to flush/compact yet.
  std::string value;
  ASSERT_TRUE(db.Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
}

TEST(Db, StatsReflectOperationCounts) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  ASSERT_TRUE(db.Put("k2", "v2").ok());
  ASSERT_TRUE(db.Put("k1", "v1-updated").ok());  // overwrite, not a new key

  std::string value;
  EXPECT_TRUE(db.Get("k1", &value).ok());   // hit
  EXPECT_TRUE(db.Get("k3", &value).IsNotFound());  // miss

  ASSERT_TRUE(db.Delete("k2").ok());

  DB::Stats stats = db.GetStats();
  EXPECT_EQ(stats.put_count, 3u);
  EXPECT_EQ(stats.delete_count, 1u);
  EXPECT_EQ(stats.get_count, 2u);
  EXPECT_EQ(stats.get_hit_count, 1u);
  EXPECT_EQ(stats.get_miss_count, 1u);
  EXPECT_EQ(stats.live_key_count, 1u);  // only k1 remains
}

TEST(Db, StatsLiveKeyCountIgnoresDeleteOfAbsentKey) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());
  ASSERT_TRUE(db.Delete("does-not-exist").ok());

  DB::Stats stats = db.GetStats();
  EXPECT_EQ(stats.live_key_count, 1u);
}

// A modest randomized-operation check against an independent std::map
// "oracle", distinct from whatever DB's in-memory mode uses internally
// (memtable::MemTableList as of Phase 4 — ADR 0013), so a bug in DB's
// Put/Delete/Get logic (as opposed to the data structure it happens to
// delegate to) would be caught. This is a light-weight precursor to,
// not a substitute for, spec §1.8's full "random operation stream vs a
// std::map reference model, across restarts" property test — the
// "across restarts" half requires real persistence, which this
// in-memory-only mode deliberately does not have; see
// tests/integration/test_db_persistence.cpp's
// RandomizedOperationsAcrossMultipleRestartsMatchOracle for that full
// version, against a real, disk-backed DB.
TEST(Db, RandomizedOperationsMatchIndependentOracle) {
  DB db;
  std::map<std::string, std::string> oracle;

  std::mt19937 rng(0xC0FFEE);
  std::uniform_int_distribution<int> op_dist(0, 2);        // put/get/delete
  std::uniform_int_distribution<int> key_dist(0, 24);      // small keyspace -> lots of overwrites
  std::uniform_int_distribution<int> val_len_dist(0, 12);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  for (int i = 0; i < 5000; ++i) {
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
      ASSERT_TRUE(db.Put(key, value).ok());
      oracle[key] = value;
    } else if (op == 1) {  // Get
      std::string actual;
      Status s = db.Get(key, &actual);
      auto it = oracle.find(key);
      if (it == oracle.end()) {
        EXPECT_TRUE(s.IsNotFound()) << "key=" << key;
      } else {
        ASSERT_TRUE(s.ok()) << "key=" << key;
        EXPECT_EQ(actual, it->second) << "key=" << key;
      }
    } else {  // Delete
      ASSERT_TRUE(db.Delete(key).ok());
      oracle.erase(key);
    }
  }

  // Final full sweep: every oracle key is present with the right value,
  // and DB's live_key_count matches the oracle's size exactly.
  for (const auto& [key, expected_value] : oracle) {
    std::string actual;
    ASSERT_TRUE(db.Get(key, &actual).ok()) << "key=" << key;
    EXPECT_EQ(actual, expected_value) << "key=" << key;
  }
  EXPECT_EQ(db.GetStats().live_key_count, oracle.size());
}

}  // namespace
}  // namespace pebbledb
