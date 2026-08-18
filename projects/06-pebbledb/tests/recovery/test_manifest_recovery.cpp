#include "pebbledb/db.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

#include "pebbledb/manifest/format.hpp"
#include "pebbledb/manifest/manifest.hpp"
#include "pebbledb/util/filenames.hpp"
#include "temp_file.hpp"

// Fault-injection coverage for DB::Flush()'s crash-safety contract
// (invariant 5, "the manifest never intentionally references a
// partially published table"; roadmap Phase 4 deliverable "manifest
// update... crash-safe updates per the stated durability model"). Each
// test here deliberately makes one specific step of FlushLocked() fail
// (by blocking the exact path it needs with a directory, so the
// underlying open()/rename() call fails deterministically) and confirms
// that no data is lost and the DB is left in a state a later Open()
// can still recover cleanly from.
//
// Normal (non-fault-injected) restart/flush behavior lives in
// tests/integration/test_db_persistence.cpp; bit-flip/truncation
// corruption of the manifest's *bytes* lives in
// tests/corruption/test_manifest_corruption.cpp.

namespace pebbledb {
namespace {

using pebbledb::testutil::TempDir;

// Blocking the SSTable file's path (the very first thing FlushLocked()
// tries to create) fails before any durable state changes at all --
// nothing to roll back, the active MemTable's data is simply still
// where it was.
TEST(ManifestRecovery, FlushFailureAtSstableCreationLeavesDataFullyRecoverable) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  ASSERT_TRUE(db->Put("k1", "v1").ok());
  ASSERT_TRUE(db->Put("k2", "v2").ok());

  // A fresh DB::Open() allocates file number 1 for its initial WAL, so
  // the first flush's SSTable is file number 2 -- block that exact path
  // by pre-occupying it with a directory (sstable::Writer::Open uses
  // util::PosixFile::OpenNew, which rejects a path that already has
  // real content, and a directory entry is exactly that).
  const std::string blocked_path = util::SstableFileName(dir.path(), 2);
  std::filesystem::create_directory(blocked_path);

  Status flush_status = db->Flush();
  EXPECT_FALSE(flush_status.ok());

  // Nothing was lost: both keys are still readable through the
  // still-frozen (but not-yet-flushed) MemTable state.
  std::string value;
  ASSERT_TRUE(db->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(db->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");

  // Unblock and retry -- FlushLocked() allocates a fresh file number for
  // the retry (the burned number 2 is never reused -- see
  // util/filenames.hpp), so this succeeds independently of the earlier
  // failure.
  std::filesystem::remove(blocked_path);
  EXPECT_TRUE(db->Flush().ok());
  EXPECT_EQ(db->GetStats().sstable_count, 1u);

  db.reset();
  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(reopened->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");
}

// Invariant 5, end to end: the SSTable file itself is successfully
// written and fsync'd, but the manifest update that would *publish* it
// fails. The table must be left referenced by nothing (an orphan on
// disk, harmless), and every key must still be recoverable -- via the
// old WAL, which a failed flush must not have deleted.
TEST(ManifestRecovery, FlushFailureAtManifestPublishLeavesTableOrphanedButDataRecoverable) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  ASSERT_TRUE(db->Put("k1", "v1").ok());
  ASSERT_TRUE(db->Put("k2", "v2").ok());

  // manifest::SaveManifest() writes to dbname/MANIFEST.tmp before
  // renaming it over dbname/MANIFEST (D7's write-new + fsync + atomic
  // rename, ADR 0012) -- block *that* path specifically, well after the
  // SSTable itself would have already been written successfully.
  const std::string blocked_temp_manifest_path = dir.path() + "/" + manifest::kManifestTempFileName;
  std::filesystem::create_directory(blocked_temp_manifest_path);

  Status flush_status = db->Flush();
  EXPECT_FALSE(flush_status.ok());

  // The SSTable itself really was written to disk...
  int sst_file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir.path())) {
    if (entry.path().extension() == ".sst") {
      ++sst_file_count;
    }
  }
  EXPECT_EQ(sst_file_count, 1) << "the table should have been written even though publish failed";

  // ...but the manifest was never updated to reference it (invariant 5).
  manifest::ManifestData on_disk_manifest;
  ASSERT_TRUE(manifest::LoadManifest(dir.path(), &on_disk_manifest).ok());
  EXPECT_TRUE(on_disk_manifest.tables.empty());

  // And nothing is lost: both keys are still readable in this same,
  // still-live DB instance (the frozen MemTable is still resident, since
  // RemoveImmutable() only runs after a successful publish).
  std::string value;
  ASSERT_TRUE(db->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(db->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");

  db.reset();  // simulated crash right after the failed Flush()
  std::filesystem::remove(blocked_temp_manifest_path);

  // A fresh Open() must recover both keys via the old WAL (never
  // deleted, since "recovered-WAL clearing" only happens after a
  // *successful* publish) -- and must not reference the orphaned
  // SSTable (the manifest on disk still lists zero tables).
  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  EXPECT_EQ(reopened->GetStats().sstable_count, 0u);
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(reopened->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");

  // The orphaned table is still sitting on disk, unreferenced -- exactly
  // the "harmless orphan" outcome ADR 0013 documents, not a correctness
  // problem, just wasted space a future compaction (Phase 6) could
  // reclaim.
  sst_file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir.path())) {
    if (entry.path().extension() == ".sst") {
      ++sst_file_count;
    }
  }
  EXPECT_EQ(sst_file_count, 1);
}

// Regression test for a real bug (ADR 0016): an earlier version of
// FlushLocked() resumed a retry from only the *oldest* still-frozen
// immutable MemTable, ignoring anything that had accumulated in the
// active MemTable since the failed attempt. Because the WAL was only
// rotated after a successful publish, a write acknowledged *between* the
// failed attempt and the successful retry ended up logged only in the
// same WAL segment the retry's success then deleted as "superseded" --
// silently losing it. This test reproduces exactly that sequence and
// confirms every acknowledged key survives a crash immediately after the
// successful retry.
TEST(ManifestRecovery, FlushRetryWithInterveningWritesLosesNoAcknowledgedData) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
  ASSERT_TRUE(db->Put("k1", "v1").ok());
  ASSERT_TRUE(db->Put("k2", "v2").ok());

  // Block the first flush attempt's SSTable path (file number 2 -- same
  // reasoning as FlushFailureAtSstableCreationLeavesDataFullyRecoverable
  // above), forcing k1/k2's freeze to succeed but the flush itself to
  // fail.
  const std::string blocked_path = util::SstableFileName(dir.path(), 2);
  std::filesystem::create_directory(blocked_path);

  Status flush_status = db->Flush();
  EXPECT_FALSE(flush_status.ok());

  // A write acknowledged *after* the failed attempt but *before* any
  // retry -- exactly the write the bug used to lose.
  ASSERT_TRUE(db->Put("k3", "v3").ok());

  std::filesystem::remove(blocked_path);
  EXPECT_TRUE(db->Flush().ok());

  std::string value;
  ASSERT_TRUE(db->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(db->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");
  ASSERT_TRUE(db->Get("k3", &value).ok());
  EXPECT_EQ(value, "v3");

  db.reset();  // simulated crash right after the successful retry
  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  ASSERT_TRUE(reopened->Get("k2", &value).ok());
  EXPECT_EQ(value, "v2");
  ASSERT_TRUE(reopened->Get("k3", &value).ok())
      << "k3 was acknowledged between the failed attempt and the successful "
         "retry -- it must survive a crash immediately after that retry";
  EXPECT_EQ(value, "v3");
}

// ADR 0016: manifest::SaveManifest()'s `published` out-param must let a
// caller distinguish "rename() never happened" (false -- the on-disk
// manifest is provably unchanged) from "rename() succeeded, only the
// trailing directory-fsync is uncertain" (true -- the on-disk manifest
// already IS the new content). Conflating those two is what let an
// earlier version of DB::FlushLocked() keep operating on stale in-memory
// state that had already been superseded on disk.
TEST(ManifestRecovery, SaveManifestPublishedIsFalseWhenRenameNeverHappens) {
  TempDir dir;
  // Block the *final* manifest path itself with a directory: rename()ing
  // a regular file (MANIFEST.tmp) onto an existing directory fails
  // outright (POSIX EISDIR) -- step 3 never succeeds.
  const std::string manifest_path = dir.path() + "/" + manifest::kManifestFileName;
  std::filesystem::create_directory(manifest_path);

  manifest::ManifestData data;
  data.next_file_number = 1;
  bool published = true;  // deliberately pre-seeded wrong, to prove SaveManifest sets it
  Status s = manifest::SaveManifest(dir.path(), data, &published);
  EXPECT_FALSE(s.ok());
  EXPECT_FALSE(published);
}

TEST(ManifestRecovery, SaveManifestPublishedIsTrueOnFullSuccess) {
  TempDir dir;
  manifest::ManifestData data;
  data.next_file_number = 1;
  bool published = false;
  Status s = manifest::SaveManifest(dir.path(), data, &published);
  EXPECT_TRUE(s.ok());
  EXPECT_TRUE(published);
}

// A fresh Open() on a brand-new directory must itself be crash-safe:
// once it returns OK, a manifest recording the newly allocated WAL must
// already be durable, so a crash immediately afterward can't reuse that
// file number for something else on the next Open().
TEST(ManifestRecovery, FreshOpenPersistsManifestBeforeReturning) {
  TempDir dir;
  std::unique_ptr<DB> db;
  ASSERT_TRUE(DB::Open(dir.path(), &db).ok());

  manifest::ManifestData on_disk_manifest;
  ASSERT_TRUE(manifest::LoadManifest(dir.path(), &on_disk_manifest).ok());
  EXPECT_NE(on_disk_manifest.wal_file_number, 0u);
  EXPECT_GT(on_disk_manifest.next_file_number, on_disk_manifest.wal_file_number);
}

}  // namespace
}  // namespace pebbledb
