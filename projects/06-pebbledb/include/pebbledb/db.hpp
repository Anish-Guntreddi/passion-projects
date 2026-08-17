#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <shared_mutex>
#include <string>

#include "pebbledb/slice.hpp"
#include "pebbledb/status.hpp"

namespace pebbledb {

// DB is PebbleDB's public key-value API (spec Part 1 §1.2: put/get/
// delete/flush/compact/stats, binary-safe keys and values).
//
// Phase 0 status (spec roadmap Phase 0, "in-memory map reference
// implementation"): DB is backed directly by an in-memory std::map, with
// no persistence and no write-ahead log wired in yet — see
// docs/architecture.md, "What's implemented so far". Flush() and
// Compact() are no-ops that return OK: there is nothing on disk yet to
// flush or compact. The method signatures below are the long-term public
// API surface and are not expected to change shape as the WAL, MemTable,
// SSTable, and compaction are wired in behind them in later phases.
//
// Concurrency model (ADR 0003, spec decision D9): single-writer,
// multi-reader. Put/Delete take DB's lock exclusively; Get/GetStats take
// it shared. This is enough to make concurrent Put/Get/Delete calls from
// multiple threads memory-safe today; it is not itself a claim about
// concurrent background work, which is out of scope until Phase 7 (spec
// Part 4 rule 3: "no complex concurrency... before the synchronous engine
// is correct and recoverable").
//
// A shared lock only excludes concurrent *exclusive* lock holders; it does
// not serialize multiple concurrent shared-lock holders against each
// other. Get()'s per-call counters are therefore atomics (see stats_/
// get_count_ below), not plain integers merely "protected" by the shared
// lock a naive reading of the mutex might suggest — see ADR 0003.
class DB {
 public:
  struct Stats {
    std::uint64_t put_count = 0;
    std::uint64_t delete_count = 0;
    std::uint64_t get_count = 0;
    std::uint64_t get_hit_count = 0;
    std::uint64_t get_miss_count = 0;
    std::size_t live_key_count = 0;
  };

  DB() = default;
  ~DB() = default;

  DB(const DB&) = delete;
  DB& operator=(const DB&) = delete;
  DB(DB&&) = delete;
  DB& operator=(DB&&) = delete;

  // Inserts or overwrites `key` with `value`. Both are copied into
  // DB-owned storage before this call returns (see slice.hpp's ownership
  // rules); the caller may destroy or mutate the memory backing `key`/
  // `value` immediately afterward. Always returns OK in Phase 0 (no
  // failure mode exists yet without persistence).
  Status Put(Slice key, Slice value);

  // Removes `key`. Deleting a key that is not present is not an error —
  // it is idempotent and returns OK, matching the tombstone semantics an
  // LSM delete will have once persistence exists (spec §1.4 FR4): a
  // delete's job is "this key must not be visible", regardless of whether
  // a value happened to be present in this specific in-memory snapshot.
  Status Delete(Slice key);

  // Looks up `key`. On success, *value is replaced with an owned copy of
  // the stored bytes and OK is returned. If `key` is absent, returns a
  // NotFound status and leaves *value unmodified.
  Status Get(Slice key, std::string* value) const;

  // No-op in Phase 0: nothing has been written to a MemTable or SSTable
  // yet, so there is nothing to snapshot/flush. Always returns OK.
  Status Flush();

  // No-op in Phase 0: there are no SSTables yet, so there is nothing to
  // compact. Always returns OK.
  Status Compact();

  Stats GetStats() const;

 private:
  mutable std::shared_mutex mutex_;
  std::map<std::string, std::string, std::less<>> table_;
  // put_count/delete_count/live_key_count are written only while holding
  // the lock exclusively (inside Put/Delete), so the shared_mutex's own
  // exclusion is sufficient synchronization for them.
  mutable Stats stats_;
  // get_count/get_hit_count/get_miss_count, by contrast, are updated by
  // Get(), which only takes the *shared* lock (by design, so concurrent
  // reads don't serialize on each other). Multiple Get() calls can run
  // that increment at the same instant, so these three need their own
  // atomicity rather than relying on the shared lock — see ADR 0003.
  // GetStats() folds their values into the Stats snapshot it returns.
  mutable std::atomic<std::uint64_t> get_count_{0};
  mutable std::atomic<std::uint64_t> get_hit_count_{0};
  mutable std::atomic<std::uint64_t> get_miss_count_{0};
};

}  // namespace pebbledb
