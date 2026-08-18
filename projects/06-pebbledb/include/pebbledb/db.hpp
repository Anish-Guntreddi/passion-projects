#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "pebbledb/cache/block_cache.hpp"
#include "pebbledb/manifest/format.hpp"
#include "pebbledb/memtable/memtable.hpp"
#include "pebbledb/memtable/memtable_list.hpp"
#include "pebbledb/slice.hpp"
#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/status.hpp"
#include "pebbledb/wal/writer.hpp"

namespace pebbledb {

// DB is PebbleDB's public key-value API (spec Part 1 §1.2: put/get/
// delete/flush/compact/stats, binary-safe keys and values).
//
// Two supported modes, both real, not one "the real DB" and one
// vestigial (ADR 0013):
//
//  - In-memory-only (the default constructor, `DB db;`): backed by
//    memtable::MemTableList alone -- no WAL, no SSTables, no manifest,
//    no persistence across a restart. This is what every one of this
//    project's Phase 0-era unit/concurrency tests exercises
//    (tests/unit/test_db_map_reference.cpp,
//    tests/unit/test_db_concurrency.cpp), and remains a legitimate,
//    supported way to use DB going forward (a scratch/testing store with
//    zero filesystem dependency), not a deprecated stepping stone.
//  - Persistent (`DB::Open(dbname, &db)`): a directory-rooted,
//    disk-backed DB wired exactly per docs/architecture.md's write/read
//    path diagram -- WAL append -> active MemTable -> (on a size
//    threshold or an explicit Flush()) immutable MemTable -> SSTable
//    file -> manifest update -> WAL rotation/cleanup. Recovers prior
//    state on Open() by loading the manifest and replaying whatever WAL
//    it points at. This is roadmap Phase 4's "DB survives restart with
//    data migrated WAL/MemTable -> SSTable" exit criterion, and where
//    Phase 5's Bloom filter (per-table, via sstable::Writer::Options)
//    and shared cache::BlockCache (D8) actually get used.
//
// persistent() reports which mode a given DB instance is in; Flush()/
// Compact() are no-ops that return OK in the in-memory mode (there is no
// disk to flush to), for real in the persistent mode (Compact() remains
// a no-op until Phase 6, per the roadmap).
//
// Concurrency model (ADR 0003, spec decision D9): single-writer,
// multi-reader. Put/Delete/Flush take DB's lock exclusively; Get/
// GetStats take it shared. This is enough to make concurrent Put/Get/
// Delete calls from multiple threads memory-safe today; it is not itself
// a claim about concurrent *background* work, which is out of scope
// until Phase 7 (spec Part 4 rule 3: "no complex concurrency... before
// the synchronous engine is correct and recoverable"). Phase 4/5's
// flush is synchronous and foreground: it runs (and holds the exclusive
// lock for its whole duration) inside whichever Put()/Delete()/Flush()
// call triggers it.
//
// A shared lock only excludes concurrent *exclusive* lock holders; it
// does not serialize multiple concurrent shared-lock holders against
// each other. Get()'s per-call counters are therefore atomics (see
// stats_/get_count_ below), not plain integers merely "protected" by the
// shared lock a naive reading of the mutex might suggest -- see ADR
// 0003.
class DB {
 public:
  struct Options {
    // Active MemTable size threshold (spec FR2's "size threshold") past
    // which Put()/Delete() trigger a synchronous flush before returning.
    // Ignored in the in-memory-only mode (nothing is ever flushed there
    // -- see class comment). Default: memtable::MemTable's own default
    // (4 MiB, ADR 0008).
    std::size_t memtable_size_threshold_bytes = memtable::MemTable::kDefaultSizeThresholdBytes;

    // WAL sync policy (spec decision D3; ADR 0006) applied to every WAL
    // segment this DB creates, including the one each flush rotates to.
    // Ignored in the in-memory-only mode (no WAL exists there).
    wal::SyncPolicy wal_sync_policy = wal::SyncPolicy::kEveryWrite;
    // Only consulted when wal_sync_policy == kBatched -- see
    // wal::Writer::Open.
    std::size_t wal_batch_size = 1;

    // Per-table Bloom filter target (spec FR3, roadmap Phase 5; ADR
    // 0014), passed straight through to every
    // sstable::Writer::Options::filter_bits_per_key a flush creates. 0
    // disables filter generation for every table this DB flushes.
    std::size_t filter_bits_per_key = 10;

    // Shared block cache capacity in bytes (spec FR7, roadmap Phase 5;
    // ADR 0015) for the cache::BlockCache this DB creates and shares
    // across every sstable::Reader it opens. 0 is a legal "cache
    // nothing" configuration (see cache::BlockCache's own doc comment),
    // not "no cache at all" -- every persistent DB always has a
    // BlockCache instance; a 0 capacity just means it never actually
    // retains anything.
    std::size_t block_cache_capacity_bytes = 8u * 1024 * 1024;  // 8 MiB
  };

  struct Stats {
    std::uint64_t put_count = 0;
    std::uint64_t delete_count = 0;
    std::uint64_t get_count = 0;
    std::uint64_t get_hit_count = 0;
    std::uint64_t get_miss_count = 0;
    // Distinct keys currently visible via the merged read path (active +
    // immutable MemTables, then every SSTable newest-first -- "newest
    // layer wins," matching Get()'s own merge). Tracked incrementally
    // (db.cpp's IsKeyCurrentlyLiveLocked(), consulted once per Put()/
    // Delete()), seeded by one full-layer scan at Recover() time -- see
    // db.cpp's ComputeLiveKeyCount() -- so GetStats() itself is O(1),
    // not a scan.
    std::size_t live_key_count = 0;
    // Persistent mode only (always 0 in-memory): number of SSTable files
    // currently referenced by the manifest, and how many synchronous
    // flushes (roadmap Phase 4) have completed successfully.
    std::size_t sstable_count = 0;
    std::uint64_t flush_count = 0;
  };

  // In-memory-only DB -- see class comment. Never fails.
  DB();
  explicit DB(const Options& options);

  // Opens (creating the directory if absent) a persistent, disk-backed
  // DB rooted at `dbname`. Recovers prior state via the manifest + WAL
  // per ADR 0013 -- roadmap Phase 4's exit criterion, "DB survives
  // restart with data migrated WAL/MemTable -> SSTable." `*dbptr` is
  // populated only on a Status::OK return.
  static Status Open(const std::string& dbname, std::unique_ptr<DB>* dbptr);
  static Status Open(const std::string& dbname, const Options& options, std::unique_ptr<DB>* dbptr);

  ~DB();

  DB(const DB&) = delete;
  DB& operator=(const DB&) = delete;
  DB(DB&&) = delete;
  DB& operator=(DB&&) = delete;

  // Inserts or overwrites `key` with `value`. Both are copied into
  // DB-owned storage before this call returns (see slice.hpp's ownership
  // rules); the caller may destroy or mutate the memory backing `key`/
  // `value` immediately afterward. In persistent mode, `key`/`value` are
  // durably WAL-appended (per Options::wal_sync_policy) before this
  // returns OK; a WAL failure is returned without applying the write to
  // the MemTable at all, so "acknowledged" and "applied" never diverge
  // (mirrors wal::Writer::AddRecord's own contract, one layer up). If
  // this Put() pushes the active MemTable over
  // Options::memtable_size_threshold_bytes, a synchronous flush runs
  // before this returns; the Put() itself has already succeeded by then
  // (it is not rolled back), but a flush failure is still returned to
  // the caller rather than silently swallowed -- see Flush().
  Status Put(Slice key, Slice value);

  // Removes `key`. Deleting a key that is not present is not an error —
  // it is idempotent and returns OK, matching the tombstone semantics
  // FR4/invariant 4 require once persistence exists: a delete's job is
  // "this key must not be visible," regardless of whether a value
  // happened to be present in this specific snapshot. Same WAL/flush
  // semantics as Put() in persistent mode.
  Status Delete(Slice key);

  // Looks up `key`. On success, *value is replaced with an owned copy of
  // the stored bytes and OK is returned. If `key` is absent (or its most
  // recent write was a Delete), returns a NotFound status and leaves
  // *value unmodified. In persistent mode, the merged read path is
  // active MemTable -> immutable MemTables, newest first -> SSTables,
  // newest first (docs/architecture.md) -- the first layer with *any*
  // record for `key` (a value or a tombstone) wins outright (invariant
  // 4). A non-OK, non-NotFound Status means a lower layer's data could
  // not be read at all (e.g. a corrupted SSTable block) -- invariant 7
  // requires this be surfaced as an error, never silently folded into
  // NotFound.
  Status Get(Slice key, std::string* value) const;

  // In-memory mode: a no-op that returns OK (there is no disk to flush
  // to -- see class comment). Persistent mode: freezes the active
  // MemTable (if non-empty) and writes every currently-pending immutable
  // MemTable -- there can be more than one if an earlier Flush() call
  // already froze one and then failed partway through, since Put()/
  // Delete() calls made in the meantime keep accumulating in a fresh
  // active MemTable rather than being lost (ADR 0016) -- each to its own
  // new SSTable file, publishing every one by atomically updating the
  // manifest (D7; invariant 5: never intentionally reference a partially
  // published table -- each manifest update only happens after that
  // table's own Writer::Finish() has already returned OK). Only once
  // every pending MemTable has been durably captured this way does this
  // rotate to a new WAL file and delete the now-superseded old WAL file
  // ("recovered-WAL clearing," roadmap Phase 4 deliverable) -- this
  // ordering (never rotate/delete the WAL until *everything* it might
  // contain is proven durable elsewhere) is what makes a flush that
  // fails, gets retried, and succeeds later never lose a write that was
  // acknowledged in between. An empty active MemTable with nothing
  // already pending makes this an OK no-op even in persistent mode. Also
  // called automatically, synchronously, from Put()/Delete() whenever
  // the active MemTable crosses Options::memtable_size_threshold_bytes
  // -- see ADR 0013 for why this project's Phase 4-6 flush is
  // synchronous/foreground rather than background (spec Part 4 rule 3;
  // background workers are Phase 7).
  Status Flush();

  // No-op that returns OK in both modes: one simple compaction strategy
  // is roadmap Phase 6, not yet implemented.
  Status Compact();

  Stats GetStats() const;

  // True if this DB is disk-backed (constructed via Open()), false for
  // the in-memory-only mode (the default constructor).
  bool persistent() const { return !dbname_.empty(); }

 private:
  DB(std::string dbname, Options options);

  // Loads the manifest (or starts fresh if none exists yet) and replays
  // whatever WAL it points at into the active MemTable, then opens a
  // wal::Writer positioned to keep appending -- called once, from
  // Open(), before a persistent DB is handed back to its caller. See
  // ADR 0013 for the full recovery algorithm.
  Status Recover();

  // Assumes mutex_ is already held exclusively. See Flush()'s doc
  // comment for what this actually does; Flush() itself is just
  // `std::unique_lock lock(mutex_); return FlushLocked();`.
  Status FlushLocked();

  // Full merge-scan across every layer (active + immutable MemTables,
  // then every open SSTable newest-first) computing the number of
  // distinct keys currently visible with a live (non-tombstone) entry.
  // O(total entries across every layer) -- called exactly once, from
  // Recover(), to establish live_key_count_'s correct starting value
  // after loading whatever manifest/WAL/SSTable state already exists;
  // never called from the Put()/Delete()/GetStats() hot path (see
  // IsKeyCurrentlyLiveLocked() and live_key_count_ below for how it
  // stays correct after that one-time seed without repeating this scan).
  // Assumes mutex_ is already held (shared is sufficient -- this only
  // reads).
  std::size_t ComputeLiveKeyCount() const;

  // Answers "does `key` currently resolve to a live (non-tombstone)
  // value via the merged read path" -- the same merge Get() itself
  // applies (active/immutable MemTables, then SSTables newest-first),
  // used by Put()/Delete() to keep live_key_count_ correct
  // incrementally without re-scanning anything. A lower-layer read
  // failure (e.g. a corrupted SSTable block) is treated as "not live"
  // here rather than propagated -- this is a best-effort stats input,
  // not a correctness-critical read, and must never block or fail the
  // write it is bookkeeping for; a real corrupted-block error is still
  // surfaced normally to any actual Get() call. Assumes mutex_ is
  // already held exclusively (called only from Put()/Delete()).
  bool IsKeyCurrentlyLiveLocked(Slice key) const;

  mutable std::shared_mutex mutex_;
  Options options_;
  std::string dbname_;  // empty => in-memory-only mode (persistent())

  memtable::MemTableList memtables_;

  // Persistent-mode-only state below (all null/empty/zero in the
  // in-memory-only mode, and never touched by Put/Delete/Get/Flush when
  // !persistent()):
  std::unique_ptr<wal::Writer> wal_writer_;
  std::uint64_t current_wal_number_ = 0;
  std::uint64_t next_file_number_ = 1;
  std::uint64_t next_sequence_ = 1;
  manifest::ManifestData manifest_;  // last durably-Save()d manifest snapshot
  std::unique_ptr<cache::BlockCache> block_cache_;
  // Open sstable::Reader per table currently referenced by manifest_,
  // oldest-flushed-first -- the same order manifest_.tables itself
  // stores them in (manifest/format.hpp) -- so index i here always
  // corresponds to manifest_.tables[i]. Get() walks this newest-first
  // (i.e. reverse iteration), matching the read-path "newest layer
  // wins" rule.
  std::vector<std::shared_ptr<sstable::Reader>> tables_;

  // Tracked incrementally by Put()/Delete() via IsKeyCurrentlyLiveLocked()
  // (mutated only under mutex_ held exclusively, matching put_count/
  // delete_count/stats_ below); seeded once, in Recover(), from
  // ComputeLiveKeyCount(). Valid in both modes: 0 for a freshly
  // default-constructed in-memory DB (nothing to seed), the real
  // recovered count for a persistent DB opened via Open().
  std::size_t live_key_count_ = 0;

  mutable Stats stats_;
  // get_count_/get_hit_count_/get_miss_count_, by contrast, are updated
  // by Get(), which only takes the *shared* lock (by design, so
  // concurrent reads don't serialize on each other). Multiple Get()
  // calls can run that increment at the same instant, so these three
  // need their own atomicity rather than relying on the shared lock —
  // see ADR 0003.
  mutable std::atomic<std::uint64_t> get_count_{0};
  mutable std::atomic<std::uint64_t> get_hit_count_{0};
  mutable std::atomic<std::uint64_t> get_miss_count_{0};
};

}  // namespace pebbledb
