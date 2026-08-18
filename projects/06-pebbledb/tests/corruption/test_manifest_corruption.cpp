#include "pebbledb/db.hpp"

#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "pebbledb/manifest/format.hpp"
#include "pebbledb/util/filenames.hpp"
#include "temp_file.hpp"

// Invariant 7 ("checksums detect corrupted/truncated records rather
// than silently returning garbage"), applied to the manifest and to
// DB::Open()'s recovery path as a whole -- the DB-level counterpart of
// tests/corruption/test_wal_corruption.cpp and
// tests/corruption/test_sstable_corruption.cpp. Pure in-memory
// DecodeManifest() corruption-status coverage (every DecodeStatus
// enumerator) lives in tests/unit/test_manifest_format.cpp; this file
// corrupts real on-disk bytes and confirms DB::Open() fails cleanly
// (a non-OK Status) instead of crashing or silently discarding data it
// cannot actually account for.

namespace pebbledb {
namespace {

using pebbledb::testutil::CorruptByteAt;
using pebbledb::testutil::TempDir;
using pebbledb::testutil::TruncateFileTo;

TEST(ManifestCorruption, OpenFailsCleanlyWhenManifestChecksumIsCorrupted) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Flush().ok());  // real manifest with one table entry now on disk
  }

  const std::string manifest_path = dir.path() + "/" + manifest::kManifestFileName;
  // Byte 10 falls within next_file_number (offset 9, 8 bytes) -- part of
  // the checksummed body, not one of the structural fields
  // (checksum/magic/format_version) checked before the checksum itself.
  CorruptByteAt(manifest_path, 10);

  std::unique_ptr<DB> reopened;
  Status s = DB::Open(dir.path(), &reopened);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
  EXPECT_EQ(reopened, nullptr);
}

TEST(ManifestCorruption, OpenFailsCleanlyWhenManifestIsTruncated) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Flush().ok());
  }

  const std::string manifest_path = dir.path() + "/" + manifest::kManifestFileName;
  TruncateFileTo(manifest_path, manifest::kHeaderSize - 1);

  std::unique_ptr<DB> reopened;
  Status s = DB::Open(dir.path(), &reopened);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

TEST(ManifestCorruption, OpenFailsCleanlyWhenManifestReferencesMissingSstableFile) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Flush().ok());
  }

  // The manifest still correctly references this table, but the file
  // itself is gone -- e.g. an external deletion, not a bug this project
  // could produce through DB's own API.
  bool removed_any = false;
  for (const auto& entry : std::filesystem::directory_iterator(dir.path())) {
    if (entry.path().extension() == ".sst") {
      std::filesystem::remove(entry.path());
      removed_any = true;
    }
  }
  ASSERT_TRUE(removed_any);

  std::unique_ptr<DB> reopened;
  Status s = DB::Open(dir.path(), &reopened);
  EXPECT_FALSE(s.ok()) << "must not silently proceed as if the referenced table never existed";
}

TEST(ManifestCorruption, OpenFailsCleanlyWhenManifestReferencesMissingWalFile) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());  // allocates + records WAL file 1
    ASSERT_TRUE(db->Put("k1", "v1").ok());
  }

  std::filesystem::remove(util::WalFileName(dir.path(), 1));

  std::unique_ptr<DB> reopened;
  Status s = DB::Open(dir.path(), &reopened);
  EXPECT_FALSE(s.ok());
}

TEST(ManifestCorruption, OpenFailsCleanlyWhenReferencedWalRecordIsCorrupted) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
  }

  // Corrupt a byte within the *first* record's fixed header (offset 10
  // falls within `sequence`, inside the checksum's coverage) -- a fully
  // present record whose bytes fail validation is wal::ReadResult::
  // kCorruption, not kTruncated (docs/durability.md's resync-on-
  // corruption policy), which DB::Recover() treats as fatal rather than
  // silently discarding this and every subsequent record without
  // telling the caller.
  CorruptByteAt(util::WalFileName(dir.path(), 1), 10);

  std::unique_ptr<DB> reopened;
  Status s = DB::Open(dir.path(), &reopened);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

// Contrast with the test above: a *torn trailing* WAL record (the
// expected shape of an unclean shutdown's last, in-flight write) is not
// an error -- Recover() must replay everything before it and succeed.
TEST(ManifestCorruption, OpenSucceedsAndRecoversPrefixWhenWalHasATornTrailingRecord) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
  }

  const std::string wal_path = util::WalFileName(dir.path(), 1);
  const auto full_size = static_cast<std::int64_t>(std::filesystem::file_size(wal_path));
  TruncateFileTo(wal_path, full_size - 1);  // tear off the last byte of the last record

  std::unique_ptr<DB> reopened;
  ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
  std::string value;
  ASSERT_TRUE(reopened->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  EXPECT_TRUE(reopened->Get("k2", &value).IsNotFound())
      << "the torn last record must not be replayed";
}

// Regression test for a real bug (ADR 0016): an earlier version of
// DB::Recover() reopened a WAL for append right after replaying past a
// torn trailing record, without first truncating away those untrusted
// bytes. A *second* unclean shutdown could then leave new, valid records
// concatenated directly after the old torn ones -- which no later replay
// can safely tell apart from a single corrupted record -- permanently
// failing DB::Open() for the whole database (not just losing the newest
// write). This test drives exactly that two-crash sequence.
TEST(ManifestCorruption, SecondUncleanShutdownAfterATornTailStillOpensCleanlyAndRecoversNewWrites) {
  TempDir dir;
  {
    std::unique_ptr<DB> db;
    ASSERT_TRUE(DB::Open(dir.path(), &db).ok());
    ASSERT_TRUE(db->Put("k1", "v1").ok());
    ASSERT_TRUE(db->Put("k2", "v2").ok());
  }  // simulated crash #1

  const std::string wal_path = util::WalFileName(dir.path(), 1);
  const auto full_size = static_cast<std::int64_t>(std::filesystem::file_size(wal_path));
  TruncateFileTo(wal_path, full_size - 1);  // tear off the last byte of k2's record

  {
    std::unique_ptr<DB> reopened;
    ASSERT_TRUE(DB::Open(dir.path(), &reopened).ok());
    std::string value;
    ASSERT_TRUE(reopened->Get("k1", &value).ok());
    EXPECT_EQ(value, "v1");
    EXPECT_TRUE(reopened->Get("k2", &value).IsNotFound());

    // A write made right after this reopen must land cleanly after the
    // (now-truncated) torn bytes, not concatenated directly after them.
    ASSERT_TRUE(reopened->Put("k3", "v3").ok());
  }  // simulated crash #2, immediately after that write

  std::unique_ptr<DB> reopened_again;
  Status s = DB::Open(dir.path(), &reopened_again);
  ASSERT_TRUE(s.ok()) << s.ToString()
                       << " -- a second unclean shutdown after a torn trailing "
                          "record must not permanently brick DB::Open()";
  std::string value;
  ASSERT_TRUE(reopened_again->Get("k1", &value).ok());
  EXPECT_EQ(value, "v1");
  EXPECT_TRUE(reopened_again->Get("k2", &value).IsNotFound());
  ASSERT_TRUE(reopened_again->Get("k3", &value).ok());
  EXPECT_EQ(value, "v3");
}

}  // namespace
}  // namespace pebbledb
