#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "pebbledb/status.hpp"
#include "pebbledb/util/posix_file.hpp"
#include "pebbledb/wal/record.hpp"

namespace pebbledb::wal {

// WAL sync policy (spec decision D3; ADR 0006). Controls *when* AddRecord
// forces an fsync, which in turn controls what "acknowledged as durable"
// means — see docs/durability.md for the exact contract of each policy
// and, importantly, what this test suite can and cannot verify about it
// (a unit test cannot simulate real kernel/power-loss data loss).
enum class SyncPolicy {
  kEveryWrite,  // fsync after every AddRecord call, before it returns OK.
  kBatched,     // fsync every `batch_size` records, or when Sync() is
                // called explicitly.
  kExplicit,    // never fsync automatically; the caller must call Sync().
};

class Writer {
 public:
  // Opens (creating if needed) the WAL file at `path` in append mode.
  // `batch_size` is only consulted when policy == kBatched and must be
  // >= 1 in that case (checked; returns InvalidArgument otherwise).
  static Status Open(const std::string& path, SyncPolicy policy, std::size_t batch_size,
                      std::unique_ptr<Writer>* out);

  ~Writer() = default;
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  // Encodes `record` (wal::EncodeRecord) and appends it to the file, then
  // applies the writer's SyncPolicy. For kEveryWrite this does not
  // return until fsync() has succeeded — i.e. AddRecord returning OK is
  // exactly the durability acknowledgment invariant §1.5 #1 refers to
  // for that policy.
  //
  // Returns Status::InvalidArgument, without writing or acknowledging
  // anything, if record.key (or, for a kPut record, record.value) exceeds
  // wal::kMaxKeyLength / kMaxValueLength — the same bounds a reader
  // enforces on replay (record.hpp). This keeps "acknowledged" and
  // "recoverable" from ever diverging for an oversized record; see
  // docs/durability.md and ADR 0006.
  Status AddRecord(const Record& record);

  // Forces an fsync of everything written so far, regardless of policy.
  // Valid (and useful) under every SyncPolicy, including kEveryWrite
  // (where it is simply redundant with the fsync AddRecord already did).
  Status Sync();

  std::uint64_t records_written() const { return records_written_; }
  std::uint64_t records_since_sync() const { return records_since_sync_; }
  std::uint64_t sync_count() const { return sync_count_; }

 private:
  Writer(std::unique_ptr<util::PosixFile> file, SyncPolicy policy, std::size_t batch_size)
      : file_(std::move(file)), policy_(policy), batch_size_(batch_size) {}

  std::unique_ptr<util::PosixFile> file_;
  SyncPolicy policy_;
  std::size_t batch_size_;
  std::uint64_t records_written_ = 0;
  std::uint64_t records_since_sync_ = 0;
  std::uint64_t sync_count_ = 0;
};

}  // namespace pebbledb::wal
