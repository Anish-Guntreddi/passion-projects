#include "pebbledb/db.hpp"

#include <filesystem>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <system_error>
#include <utility>

#include "pebbledb/entry_type.hpp"
#include "pebbledb/lookup_result.hpp"
#include "pebbledb/manifest/manifest.hpp"
#include "pebbledb/sstable/writer.hpp"
#include "pebbledb/util/filenames.hpp"
#include "pebbledb/util/posix_file.hpp"
#include "pebbledb/wal/reader.hpp"

namespace pebbledb {

DB::DB() : dbname_() {}

DB::DB(const Options& options) : options_(options), dbname_() {}

DB::DB(std::string dbname, Options options) : options_(std::move(options)), dbname_(std::move(dbname)) {
  block_cache_ = std::make_unique<cache::BlockCache>(options_.block_cache_capacity_bytes);
}

DB::~DB() = default;

Status DB::Open(const std::string& dbname, std::unique_ptr<DB>* dbptr) {
  return Open(dbname, Options{}, dbptr);
}

Status DB::Open(const std::string& dbname, const Options& options, std::unique_ptr<DB>* dbptr) {
  std::error_code ec;
  std::filesystem::create_directories(dbname, ec);
  if (ec) {
    return Status::IOError("failed to create DB directory '" + dbname + "': " + ec.message());
  }

  std::unique_ptr<DB> db(new DB(dbname, options));
  Status s = db->Recover();
  if (!s.ok()) {
    return s;
  }
  *dbptr = std::move(db);
  return Status::OK();
}

Status DB::Recover() {
  manifest::ManifestData loaded;
  Status manifest_status = manifest::LoadManifest(dbname_, &loaded);
  if (manifest_status.ok()) {
    manifest_ = loaded;
  } else if (!manifest_status.IsNotFound()) {
    // A manifest exists but failed to decode -- see manifest.hpp's
    // LoadManifest doc comment for why this project deliberately fails
    // Open() outright here rather than attempting to auto-recover.
    return manifest_status;
  }
  // else: no manifest yet -- manifest_ stays default-constructed
  // (next_file_number=1, wal_file_number=0, last_sequence=0, no tables),
  // i.e. "start fresh."

  next_file_number_ = manifest_.next_file_number;

  // Open every table already recorded in the manifest, in the same
  // oldest-flushed-first order manifest_.tables itself stores them in
  // (manifest/format.hpp) -- tables_[i] corresponds to
  // manifest_.tables[i]. Sharing block_cache_ (keyed by file number,
  // ADR 0015) across every one of these Readers is exactly D8's "cache
  // key/ownership" default.
  tables_.clear();
  tables_.reserve(manifest_.tables.size());
  for (const manifest::TableFileInfo& table : manifest_.tables) {
    std::unique_ptr<sstable::Reader> reader;
    Status ts = sstable::Reader::Open(util::SstableFileName(dbname_, table.file_number), &reader,
                                       block_cache_.get(), table.file_number);
    if (!ts.ok()) {
      return ts;
    }
    tables_.push_back(std::shared_ptr<sstable::Reader>(reader.release()));
  }

  std::uint64_t max_sequence_seen = manifest_.last_sequence;

  if (manifest_.wal_file_number != 0) {
    // Replay the WAL the manifest says must still be applied on top of
    // the table set above, into the (currently empty) active MemTable --
    // exactly the "recover prior state" half of roadmap Phase 4's exit
    // criterion.
    std::unique_ptr<wal::Reader> wal_reader;
    Status rs = wal::Reader::Open(util::WalFileName(dbname_, manifest_.wal_file_number), &wal_reader);
    if (!rs.ok()) {
      return rs;
    }
    wal::Reader::ReplayResult replay = wal_reader->Replay([&](const wal::Record& record) {
      if (record.sequence > max_sequence_seen) {
        max_sequence_seen = record.sequence;
      }
      if (record.type == wal::RecordType::kPut) {
        memtables_.Put(record.key, record.value, record.sequence);
      } else {
        memtables_.Delete(record.key, record.sequence);
      }
    });
    // kEndOfLog (a clean read) and kTruncated (a torn trailing record --
    // the expected shape of an unclean shutdown's last, in-flight write,
    // docs/durability.md) both mean recovery completed normally: every
    // record before the truncation point replayed. kCorruption/
    // kUnsupportedVersion mean bytes that were fully, successfully
    // written are corrupted -- treated as fatal here, consistent with
    // the WAL's own no-resynchronization design (ADR 0006) generalized
    // one layer up, rather than silently discarding whatever came after
    // the corruption point without telling the caller.
    if (replay.terminal == wal::ReadResult::kCorruption ||
        replay.terminal == wal::ReadResult::kUnsupportedVersion) {
      return Status::Corruption("WAL " + std::to_string(manifest_.wal_file_number) +
                                 " corrupted during replay: " + wal::ReadResultName(replay.terminal));
    }
    if (replay.terminal == wal::ReadResult::kTruncated) {
      // A torn trailing record is non-fatal (the expected shape of an
      // unclean shutdown), but its bytes are not trusted -- everything
      // from wal_reader's valid_prefix_length() onward must be discarded
      // *before* this file is ever reopened for append below. Without
      // this, a second unclean shutdown could leave new, valid records
      // concatenated directly after these torn bytes; a later replay
      // cannot tell that apart from a single corrupted record and would
      // fail Open() permanently -- fixed here per ADR 0016 (a real bug,
      // reproduced during review: two unclean shutdowns in a row could
      // permanently brick DB::Open() for the whole database).
      Status ts = util::TruncateFile(util::WalFileName(dbname_, manifest_.wal_file_number),
                                      wal_reader->valid_prefix_length());
      if (!ts.ok()) {
        return ts;
      }
    }
    wal_reader.reset();  // close the read handle before reopening for append below

    std::unique_ptr<wal::Writer> writer;
    Status ws = wal::Writer::Open(util::WalFileName(dbname_, manifest_.wal_file_number),
                                   options_.wal_sync_policy, options_.wal_batch_size, &writer);
    if (!ws.ok()) {
      return ws;
    }
    wal_writer_ = std::move(writer);
    current_wal_number_ = manifest_.wal_file_number;
  } else {
    // Fresh DB (no manifest existed yet): allocate and create a
    // brand-new WAL, then immediately persist a manifest that records
    // it, so a crash right after this Open() call cannot leave an
    // allocated-but-unrecorded file number that a later Open() might
    // reuse for something else (ADR 0013).
    current_wal_number_ = next_file_number_++;
    std::unique_ptr<wal::Writer> writer;
    Status ws = wal::Writer::Open(util::WalFileName(dbname_, current_wal_number_),
                                   options_.wal_sync_policy, options_.wal_batch_size, &writer);
    if (!ws.ok()) {
      return ws;
    }
    wal_writer_ = std::move(writer);

    manifest_.next_file_number = next_file_number_;
    manifest_.wal_file_number = current_wal_number_;
    manifest_.last_sequence = max_sequence_seen;  // 0 for a genuinely fresh DB
    // No `published` out-param needed here (contrast FlushLocked(), ADR
    // 0016): manifest_ already holds exactly the data this call is trying
    // to persist regardless of outcome, and on any failure this whole
    // Open() call fails -- there is no "keep running with stale in-memory
    // state" risk to guard against. A caller that retries Open() after a
    // failure here simply re-runs Recover(): if the rename actually did
    // succeed (only the trailing directory-fsync failed), LoadManifest()
    // now finds it and this whole fresh-DB branch is skipped in favor of
    // ordinary (non-fresh) recovery -- still correct either way.
    Status ss = manifest::SaveManifest(dbname_, manifest_);
    if (!ss.ok()) {
      return ss;
    }
  }

  next_sequence_ = max_sequence_seen + 1;
  // One-time seed for live_key_count_'s incremental tracking (see
  // db.hpp) -- everything after this point keeps it correct via
  // IsKeyCurrentlyLiveLocked() in Put()/Delete(), without repeating this
  // full scan.
  live_key_count_ = ComputeLiveKeyCount();
  return Status::OK();
}

Status DB::Put(Slice key, Slice value) {
  std::unique_lock lock(mutex_);
  const std::uint64_t sequence = next_sequence_;

  if (persistent()) {
    wal::Record record{sequence, wal::RecordType::kPut, std::string(key), std::string(value)};
    Status s = wal_writer_->AddRecord(record);
    if (!s.ok()) {
      // Not acknowledged, not applied -- the WAL record is what makes a
      // write durable/recoverable at all, so a write that never reached
      // it must not silently become visible in the MemTable either
      // (mirrors wal::Writer::AddRecord's own "acknowledged and
      // recoverable never diverge" contract, one layer up).
      return s;
    }
  }
  ++next_sequence_;

  const bool was_live = IsKeyCurrentlyLiveLocked(key);
  memtables_.Put(key, value, sequence);
  if (!was_live) {
    ++live_key_count_;
  }
  ++stats_.put_count;

  if (persistent() && memtables_.ActiveShouldFlush(options_.memtable_size_threshold_bytes)) {
    // Put()'s own effect above has already happened and is not rolled
    // back by a subsequent flush failure -- but the failure is still
    // reported, rather than swallowed, so a caller cannot miss that the
    // DB is now behind on flushing an over-threshold MemTable (see
    // db.hpp's Put() doc comment and ADR 0013).
    Status s = FlushLocked();
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status DB::Delete(Slice key) {
  std::unique_lock lock(mutex_);
  const std::uint64_t sequence = next_sequence_;

  if (persistent()) {
    wal::Record record{sequence, wal::RecordType::kDelete, std::string(key), std::string()};
    Status s = wal_writer_->AddRecord(record);
    if (!s.ok()) {
      return s;
    }
  }
  ++next_sequence_;

  const bool was_live = IsKeyCurrentlyLiveLocked(key);
  memtables_.Delete(key, sequence);
  if (was_live) {
    --live_key_count_;
  }
  ++stats_.delete_count;

  if (persistent() && memtables_.ActiveShouldFlush(options_.memtable_size_threshold_bytes)) {
    Status s = FlushLocked();
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status DB::Get(Slice key, std::string* value) const {
  std::shared_lock lock(mutex_);
  get_count_.fetch_add(1, std::memory_order_relaxed);

  std::uint64_t sequence = 0;
  LookupResult result = memtables_.Get(key, value, &sequence);

  if (result == LookupResult::kNotFound) {
    // Newest SSTable first (tables_ is stored oldest-first, matching
    // manifest_.tables' own on-disk order -- see manifest/format.hpp and
    // db.hpp) -- the first table with *any* record for `key` (a value or
    // a tombstone) wins outright, exactly the "newest layer wins" rule
    // docs/architecture.md's read-path diagram applies at every layer.
    for (auto it = tables_.rbegin(); it != tables_.rend(); ++it) {
      Status s = (*it)->Get(key, &result, value, &sequence);
      if (!s.ok()) {
        // invariant 7: a lower layer's corruption/IO failure must be
        // surfaced as an error, never silently folded into "not found."
        return s;
      }
      if (result != LookupResult::kNotFound) {
        break;
      }
    }
  }

  if (result == LookupResult::kFound) {
    get_hit_count_.fetch_add(1, std::memory_order_relaxed);
    return Status::OK();
  }
  get_miss_count_.fetch_add(1, std::memory_order_relaxed);
  return Status::NotFound();
}

Status DB::Flush() {
  std::unique_lock lock(mutex_);
  return FlushLocked();
}

Status DB::FlushLocked() {
  if (!persistent()) {
    return Status::OK();  // nothing on disk to flush to -- see db.hpp
  }

  // Freeze whatever is currently active, if anything -- including
  // whatever Put()/Delete() accumulated there since a *previous* failed
  // Flush() call already froze (and failed to publish) something else.
  // This is the root-cause fix for a real bug (ADR 0016): an earlier
  // version of this method only froze when memtables_.ImmutableCount()
  // == 0, then resumed a retry from just the *oldest* immutable,
  // ignoring anything newer that had accumulated in the active MemTable
  // in the meantime. Because the WAL was (and still is, below) only
  // rotated once *all* pending data has been published, those
  // intervening writes were logged into the very same WAL segment the
  // retry's eventual success would then delete as "superseded" --
  // silently losing any write acknowledged between the failed attempt
  // and the successful retry. Freezing unconditionally here instead
  // means every call works from a complete picture of everything not yet
  // durably captured in an SSTable, whether that arrived via one prior
  // failed attempt or several.
  if (!memtables_.active().empty()) {
    memtables_.Freeze();
  }
  if (memtables_.ImmutableCount() == 0) {
    return Status::OK();  // nothing pending anywhere
  }

  // Flush every currently-pending immutable MemTable, oldest first (back
  // of the newest-first deque -- memtable_list.hpp), each into its own
  // SSTable + manifest publish. Because mutex_ is held exclusively for
  // this entire call, nothing new can be written in between iterations,
  // so by the time this loop drains to ImmutableCount() == 0 every write
  // that was ever logged into the *current* WAL segment (current_wal_number_,
  // unrotated so far) is guaranteed durably captured in a published
  // SSTable -- only then, below, is it safe to rotate to a new WAL and
  // delete this one.
  while (memtables_.ImmutableCount() > 0) {
    const std::shared_ptr<memtable::MemTable> frozen = memtables_.immutables().back();

    const std::uint64_t table_number = next_file_number_++;
    const std::string table_path = util::SstableFileName(dbname_, table_number);

    sstable::Writer::Options writer_options;
    writer_options.filter_bits_per_key = options_.filter_bits_per_key;

    std::unique_ptr<sstable::Writer> writer;
    Status s = sstable::Writer::Open(table_path, writer_options, &writer);
    if (!s.ok()) {
      // `frozen` remains in memtables_.immutables() -- its data is still
      // fully readable via the merged Get() path and still fully present
      // in the still-intact current WAL, so nothing is lost; the caller
      // just needs to retry Flush() (or another auto-triggered flush)
      // later. Nothing below this point has happened yet, so there is
      // nothing to roll back.
      return s;
    }

    for (const auto& [key, entry] : frozen->entries()) {
      Status add_status = writer->Add(key, entry.value, entry.sequence, entry.type);
      if (!add_status.ok()) {
        return add_status;
      }
    }
    s = writer->Finish();
    if (!s.ok()) {
      return s;
    }

    // The table file is now durably complete (Writer::Finish()'s fsync
    // guarantee) -- only *now* is it safe to publish it via the manifest
    // (invariant 5: the manifest never intentionally references a
    // partially published table).
    std::unique_ptr<sstable::Reader> reader;
    s = sstable::Reader::Open(table_path, &reader, block_cache_.get(), table_number);
    if (!s.ok()) {
      return s;
    }

    manifest::ManifestData new_manifest = manifest_;
    new_manifest.next_file_number = next_file_number_;
    new_manifest.wal_file_number = current_wal_number_;  // not rotated yet -- see below
    new_manifest.last_sequence = next_sequence_ - 1;
    new_manifest.tables.push_back(manifest::TableFileInfo{table_number, /*level=*/0});

    bool published = false;
    s = manifest::SaveManifest(dbname_, new_manifest, &published);
    if (!s.ok() && !published) {
      // The rename itself never happened (see manifest.hpp's
      // SaveManifest() doc comment) -- the on-disk manifest is provably
      // unchanged. The new SSTable exists on disk but is referenced by
      // nothing -- safe: a future Open()/Flush() simply leaves it as a
      // harmless orphan (ADR 0013). `frozen` is still in
      // memtables_.immutables() and the current WAL is untouched, so no
      // data is at risk; the caller can retry.
      return s;
    }
    // Either the publish fully succeeded, or (ADR 0016) the rename
    // succeeded and only the trailing directory-fsync failed -- either
    // way, `new_manifest` IS now dbname/MANIFEST's on-disk content, so it
    // must be adopted here rather than left stale: continuing to serve
    // out of the *old* manifest_ while the file on disk already points
    // at `new_manifest` is exactly the kind of live-process/on-disk
    // divergence that could reintroduce a lost-write bug through a third
    // door if this process crashed later while still "disagreeing" with
    // its own manifest file.
    manifest_ = std::move(new_manifest);
    tables_.push_back(std::shared_ptr<sstable::Reader>(reader.release()));
    memtables_.RemoveImmutable(frozen);
    ++stats_.flush_count;
    if (!s.ok()) {
      // Report the degraded outcome (this publish's directory-fsync
      // durability is not guaranteed) rather than silently swallowing
      // it, even though the data itself is safe and already adopted.
      return s;
    }
  }

  // Every immutable MemTable that existed at any point during this call
  // (including ones frozen by an earlier, failed attempt) is now durably
  // captured in a published SSTable, and mutex_ has been held exclusively
  // throughout, so nothing further could have been appended to the
  // current WAL since. Only now is it safe to rotate to a new WAL and
  // delete the old one.
  const std::uint64_t new_wal_number = next_file_number_++;
  std::unique_ptr<wal::Writer> new_wal_writer;
  Status ws = wal::Writer::Open(util::WalFileName(dbname_, new_wal_number), options_.wal_sync_policy,
                                 options_.wal_batch_size, &new_wal_writer);
  if (!ws.ok()) {
    // Every table produced by the loop above is already safely durable
    // and correctly referenced by the manifest; only the WAL rotation
    // itself failed. The old WAL is untouched and still correctly
    // referenced by manifest_.wal_file_number, so nothing is at risk --
    // the caller can simply retry Flush() again later (the loop above
    // will find ImmutableCount() == 0 and fall straight through to
    // retrying just this rotation).
    return ws;
  }

  manifest::ManifestData rotated_manifest = manifest_;
  rotated_manifest.wal_file_number = new_wal_number;

  bool rotate_published = false;
  Status rs = manifest::SaveManifest(dbname_, rotated_manifest, &rotate_published);
  if (!rs.ok() && !rotate_published) {
    return rs;
  }

  // Same adopt-even-if-degraded reasoning as the per-table publish above
  // (ADR 0016): once rename() has succeeded, dbname/MANIFEST already
  // points at new_wal_number on disk, so this process must adopt that
  // too, rather than keep appending to a WAL its own manifest file no
  // longer references.
  const std::string old_wal_path = util::WalFileName(dbname_, current_wal_number_);
  manifest_ = std::move(rotated_manifest);
  wal_writer_ = std::move(new_wal_writer);
  current_wal_number_ = new_wal_number;

  // "Recovered-WAL clearing" (roadmap Phase 4 deliverable): the old
  // WAL's data is now durably captured in the SSTable(s) just published,
  // so it is no longer needed for recovery -- delete it. Best-effort: a
  // failure to delete leaves a harmless orphan file (the manifest no
  // longer points at it, so nothing ever reads it again), not a
  // correctness problem, so it is not propagated as a Flush() failure.
  std::error_code ec;
  std::filesystem::remove(old_wal_path, ec);

  if (!rs.ok()) {
    return rs;  // degraded (directory-fsync failed) but fully adopted/safe
  }
  return Status::OK();
}

Status DB::Compact() { return Status::OK(); }

std::size_t DB::ComputeLiveKeyCount() const {
  // "First (newest) layer to mention a key wins" -- try_emplace only
  // inserts if the key is not already present, so processing layers
  // newest-to-oldest and recording each key's *first* sighting
  // correctly implements the same "newest layer wins outright" merge
  // Get() itself applies, without needing to compare sequence numbers
  // across layers (each individual layer has already collapsed itself
  // to one entry per key -- see memtable_list.hpp).
  std::map<std::string, bool, std::less<>> resolved;
  auto record = [&resolved](const std::string& key, bool live) { resolved.try_emplace(key, live); };

  for (const auto& [key, entry] : memtables_.active().entries()) {
    record(key, entry.type == EntryType::kValue);
  }
  for (const auto& immutable : memtables_.immutables()) {
    for (const auto& [key, entry] : immutable->entries()) {
      record(key, entry.type == EntryType::kValue);
    }
  }
  for (auto it = tables_.rbegin(); it != tables_.rend(); ++it) {
    // Best-effort: a corrupted/unreadable table simply stops
    // contributing further entries to this stats-only scan rather than
    // failing GetStats() outright (which has no Status to report through
    // -- see db.hpp's Stats::live_key_count doc comment).
    (*it)->ForEachEntry([&record](const sstable::Entry& entry) {
      record(entry.key, entry.type == EntryType::kValue);
    });
  }

  std::size_t live = 0;
  for (const auto& [key, is_live] : resolved) {
    if (is_live) {
      ++live;
    }
  }
  return live;
}

bool DB::IsKeyCurrentlyLiveLocked(Slice key) const {
  std::string discard_value;
  std::uint64_t discard_sequence = 0;
  LookupResult result = memtables_.Get(key, &discard_value, &discard_sequence);

  if (result == LookupResult::kNotFound) {
    for (auto it = tables_.rbegin(); it != tables_.rend(); ++it) {
      Status s = (*it)->Get(key, &result, &discard_value, &discard_sequence);
      if (!s.ok()) {
        // See this method's doc comment (db.hpp): a lower-layer read
        // failure here must not propagate -- this is bookkeeping for a
        // stat, not the write itself.
        return false;
      }
      if (result != LookupResult::kNotFound) {
        break;
      }
    }
  }
  return result == LookupResult::kFound;
}

DB::Stats DB::GetStats() const {
  std::shared_lock lock(mutex_);
  Stats snapshot = stats_;
  snapshot.get_count = get_count_.load(std::memory_order_relaxed);
  snapshot.get_hit_count = get_hit_count_.load(std::memory_order_relaxed);
  snapshot.get_miss_count = get_miss_count_.load(std::memory_order_relaxed);
  snapshot.sstable_count = tables_.size();
  snapshot.live_key_count = live_key_count_;
  return snapshot;
}

}  // namespace pebbledb
