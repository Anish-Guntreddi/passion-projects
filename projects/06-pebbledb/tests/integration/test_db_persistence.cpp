#include "pebbledb/db.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "temp_file.hpp"

// Roadmap Phase 4 exit criterion: "DB survives restart with data
// migrated WAL/MemTable -> SSTable." Every test here opens a real,
// disk-backed pebbledb::DB (DB::Open()) rooted at a TempDir, destroys
// the DB object with no special shutdown step (the same "simulated
// crash/restart" pattern tests/recovery and tests/integration use
// elsewhere in this project -- nothing survives in-process except what
// actually landed on disk), and opens a brand-new DB on the same
// directory to confirm state was recovered correctly. Fault-injection
// (a flush that fails partway through) lives in
// tests/recovery/test_manifest_recovery.cpp; bit-flip/truncation
// corruption of the manifest itself lives in
// tests/corruption/test_manifest_corruption.cpp.

namespace pebbledb {
namespace {

using pebbledb::testutil::TempDir;

TEST(DbPersistence, PutGetDeleteWorkWithoutAnyRestart) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  EXPECT_TRUE(db->persistent());

  ASSERT_TRUE(db->Put("k1", "v1").ok());
  std::string value;
  ASSERT_TRUE(db->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");

  ASSERT_TRUE(db->Delete("k1").ok());
  EXPECT_TRUE(db->Get("k1", &value).IsNotFound());
}

TEST(DbPersistence, DataWrittenButNeverFlushedSurvivesRestartViaWalReplay) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
    ASSERT_TRUE(db->Put("k3", "v3").ok());
    EXPECT_EQ(db->GetStats().sstable_count, 0u);  // nothing flushed yet
  }  // `db` destroyed -- simulated restart; only the WAL is on disk

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  std::string value;
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(reopened->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");
  ASSERT_TRUE(reopened->Get("k3", &value).ok());
  EXPECT_EQ(value, "v3");
}

TEST(DbPersistence, DataSurvivesRestartAfterExplicitFlushViaSstable) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
    ASSERT_TRUE(db->Flush().ok());
    EXPECT_EQ(db->GetStats().sstable_count, 1u);
    EXPECT_EQ(db->GetStats().flush_count, 1u);
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  EXPECT_EQ(reopened->GetStats().sstable_count, 1u);
  std::string value;
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(reopened->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");
}

TEST(DbPersistence, FlushOnEmptyMemtableIsOkNoOp) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  EXPECT_TRUE(db->Flush().ok());
  EXPECT_EQ(db->GetStats().sstable_count, 0u);
  EXPECT_EQ(db->GetStats().flush_count, 0u);
}

TEST(DbPersistence, ReopeningFreshEmptyDirectoryStartsWithNoData) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  DB::Stats stats = db->GetStats();
  EXPECT_EQ(stats.put_count, 0u);
  EXPECT_EQ(stats.sstable_count, 0u);
  std::string value;
  EXPECT_TRUE(db->Get("anything", &value).IsNotFound());
}

// Automatic, threshold-triggered flushing (spec FR2/roadmap Phase 4) --
// not just an explicit Flush() call.
TEST(DbPersistence, AutomaticFlushOnSizeThresholdProducesMultipleSstables) {
  TempDir dir;
  DB::Options options;
  options.memtable_size_threshold_bytes = 512;  // tiny, to force several flushes

  std::map<std::string, std::string> oracle;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), options, &db).ok());
    for (int i = 0; i < 200; ++i) {
      std::string key = "key" + std::to_string(i);
      std::string value = "value-" + std::to_string(i) + "-" + std::string(20, 'x');
      ASSERT_TRUE(db->Put(key, value).ok());
      oracle[key] = value;
    }
    EXPECT_GT(db->GetStats().sstable_count, 1u) << "threshold should have forced multiple flushes";
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), options, &reopened).ok());
  for (const auto& [key, expected] : oracle) {
    std::string actual;
    ASSERT_TRUE(reopened->Get(key, &actual).ok()) << "key=" << key;
    EXPECT_EQ(actual, expected) << "key=" << key;
  }
}

// Invariant 4 (spec §1.5): a tombstone recorded only in the WAL (never
// flushed) still shadows an older, already-flushed value after restart.
TEST(DbPersistence, Invariant4_TombstoneAfterFlushShadowsOlderFlushedValueAcrossRestart) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "old-value").ok());
    ASSERT_TRUE(db->Flush().ok());  // "old-value" now lives in an SSTable
    ASSERT_TRUE(db->Delete("k1").ok());  // tombstone recorded only in the (new) WAL
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  std::string value;
  EXPECT_TRUE(reopened->Get("k1", &value).IsNotFound())
      << "stale flushed value must not resurface past a later tombstone";
}

// Same invariant, but with the tombstone *also* flushed to its own,
// newer SSTable -- the two-tables-newest-wins case.
TEST(DbPersistence, Invariant4_TombstoneInNewerSstableShadowsValueInOlderSstable) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "old-value").ok());
    ASSERT_TRUE(db->Flush().ok());
    ASSERT_TRUE(db->Delete("k1").ok());
    ASSERT_TRUE(db->Flush().ok());
    EXPECT_EQ(db->GetStats().sstable_count, 2u);
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  std::string value;
  EXPECT_TRUE(reopened->Get("k1", &value).IsNotFound());
}

// Overwrite (not delete) across two flushed SSTables: the newer table's
// value must win, both before and after a restart.
TEST(DbPersistence, OverwriteAcrossTwoFlushedSstablesResolvesToNewestValue) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Flush().ok());
    ASSERT_TRUE(db->Put("k1", "v2").ok());
    ASSERT_TRUE(db->Flush().ok());

    std::string value;
    ASSERT_TRUE(db->Get("k1", &value).ok());
    EXPECT_EQ(value, "v2");
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  std::string value;
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v2");
}

// "Recovered-WAL clearing" (roadmap Phase 4 deliverable): a successful
// flush deletes the WAL segment it superseded.
TEST(DbPersistence, SuccessfulFlushDeletesTheSupersededWalFile) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  ASSERT_TRUE(db->Put("k1", "v1").ok());

  auto CountFilesWithExtension = [&dir](const std::string& extension) {
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir.path())) {
      if (entry.path().extension() == extension) {
        ++count;
      }
    }
    return count;
  };

  EXPECT_EQ(CountFilesWithExtension(".wal"), 1);
  EXPECT_EQ(CountFilesWithExtension(".sst"), 0);

  ASSERT_TRUE(db->Flush().ok());

  // The old WAL is gone (deleted), a new (empty, current) one exists,
  // and the flushed data now lives in exactly one SSTable.
  EXPECT_EQ(CountFilesWithExtension(".wal"), 1);
  EXPECT_EQ(CountFilesWithExtension(".sst"), 1);
}

TEST(DbPersistence, ManifestFileExistsAfterFirstOpenEvenWithNoWrites) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  EXPECT_TRUE(std::filesystem::exists(dir.path() + "/MANIFEST"));
}

TEST(DbPersistence, LiveKeyCountAcrossFlushAndRestartMatchesExpectedDistinctKeys) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
    ASSERT_TRUE(db->Flush().ok());
    ASSERT_TRUE(db->Put("k3", "v3").ok());  // stays in the active MemTable
    ASSERT_TRUE(db->Delete("k1").ok());     // tombstone, also unflushed

    EXPECT_EQ(db->GetStats().live_key_count, 2u);  // k2 (flushed) + k3 (active); k1 deleted
  }

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  EXPECT_EQ(reopened->GetStats().live_key_count, 2u);
}

// spec §1.8: "random operation stream vs a std::map reference model,
// across restarts." Periodically flushes and reopens the DB (a real
// restart, not merely an in-process Flush() call) and checks every
// key against an independent std::map oracle both mid-stream and at
// the end.
TEST(DbPersistence, RandomizedOperationsAcrossMultipleRestartsMatchOracle) {
  TempDir dir;
  DB::Options options;
  options.memtable_size_threshold_bytes = 2048;  // force some automatic flushes too

  std::map<std::string, std::string> oracle;
  std::mt19937 rng(0xFEEDFACE);
  std::uniform_int_distribution<int> op_dist(0, 2);   // put/get/delete
  std::uniform_int_distribution<int> key_dist(0, 39);
  std::uniform_int_distribution<int> val_len_dist(0, 24);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), options, &db).ok());

  constexpr int kTotalOps = 4000;
  constexpr int kRestartEveryOps = 400;
  for (int i = 0; i < kTotalOps; ++i) {
    if (i > 0 && i % kRestartEveryOps == 0) {
      db.reset();  // simulated restart
      ASSERT_TRUE(DB::Open(dir.path(), options, &db).ok());
      for (const auto& [key, expected_value] : oracle) {
        std::string actual;
        ASSERT_TRUE(db->Get(key, &actual).ok()) << "post-restart key=" << key;
        EXPECT_EQ(actual, expected_value) << "post-restart key=" << key;
      }
    }

    std::ostringstream key_stream;
    key_stream << "key" << key_dist(rng);
    std::string key = key_stream.str();

    const int op = op_dist(rng);
    if (op == 0) {  // Put
      std::string value;
      const int len = val_len_dist(rng);
      for (int b = 0; b < len; ++b) {
        value.push_back(static_cast<char>(static_cast<unsigned char>(byte_dist(rng))));
      }
      ASSERT_TRUE(db->Put(key, value).ok());
      oracle[key] = value;
    } else if (op == 1) {  // Get
      std::string actual;
      Status s = db->Get(key, &actual);
      auto it = oracle.find(key);
      if (it == oracle.end()) {
        EXPECT_TRUE(s.IsNotFound()) << "key=" << key;
      } else {
        ASSERT_TRUE(s.ok()) << "key=" << key;
        EXPECT_EQ(actual, it->second) << "key=" << key;
      }
    } else {  // Delete
      ASSERT_TRUE(db->Delete(key).ok());
      oracle.erase(key);
    }
  }

  // Final restart, then a full sweep of the oracle.
  db.reset();
  ASSERT_TRUE(DB::Open(dir.path(), options, &db).ok());
  for (const auto& [key, expected_value] : oracle) {
    std::string actual;
    ASSERT_TRUE(db->Get(key, &actual).ok()) << "final key=" << key;
    EXPECT_EQ(actual, expected_value) << "final key=" << key;
  }
  EXPECT_EQ(db->GetStats().live_key_count, oracle.size());
}

}  // namespace
}  // namespace pebbledb
