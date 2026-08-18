#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace pebbledb::cache {

// BlockCache: an LRU cache of decoded SSTable data-block contents,
// shared across every open sstable::Reader in a DB (spec FR7, roadmap
// Phase 5; decision D8's default: key = (file_id, block_offset), the
// cache owns its own copy of each cached block rather than aliasing a
// Reader's buffer). Bounded by total cached bytes (`capacity_bytes`),
// not entry count, since blocks vary in size (a single entry can exceed
// ADR 0009's *target* block size).
//
// What is cached: the *post-verification, trailer-stripped* entries
// bytes of a data block -- i.e. exactly what sstable::Reader::Get()/
// ForEachEntry() need to decode, not the raw on-disk bytes (checksum +
// compression-type trailer already stripped and verified once, at
// insert time, so a cache hit skips re-verification too, not just the
// pread). Index/filter/footer regions are not cached here -- each is
// already loaded once into the owning Reader's own memory at Open() and
// never re-read, so there is nothing repeated to cache for them.
//
// Thread-safety: every public method takes an internal mutex --
// multiple sstable::Reader objects, and multiple concurrent
// DB::Get() calls (which only take DB's *shared* lock -- ADR 0003), can
// Lookup()/Insert() concurrently. See ADR 0015.
class BlockCache {
 public:
  struct Stats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t inserts = 0;
    std::uint64_t evictions = 0;
    std::size_t current_bytes = 0;
  };

  // `capacity_bytes == 0` is a legal, degenerate "cache nothing" cache:
  // every Insert() immediately evicts whatever it just inserted (see
  // Insert()'s comment) and every Lookup() misses -- useful for an
  // apples-to-apples "cache off" comparison against a nonzero-capacity
  // cache without a separate on/off code path in sstable::Reader.
  explicit BlockCache(std::size_t capacity_bytes);

  BlockCache(const BlockCache&) = delete;
  BlockCache& operator=(const BlockCache&) = delete;

  // Looks up (`file_id`, `block_offset`). On a hit, `*out` is set to an
  // owned copy of the cached block bytes and the entry is promoted to
  // most-recently-used; returns true. On a miss, returns false and
  // `*out` is left unmodified.
  bool Lookup(std::uint64_t file_id, std::uint64_t block_offset, std::string* out);

  // Inserts (or overwrites) the cache entry for (`file_id`,
  // `block_offset`) with an owned copy of `block`. Evicts
  // least-recently-used entries (oldest first) until the cache is back
  // at or under `capacity_bytes` -- an invariant this class maintains
  // unconditionally: current_bytes() never exceeds capacity_bytes()
  // after any call returns. A single block larger than the entire cache
  // capacity is still inserted (never rejected outright), but then --
  // since it alone already exceeds capacity -- immediately evicted again
  // by that same trim step, along with every other entry; the net
  // effect is that this Insert() call leaves the cache empty rather than
  // "caching" something it structurally cannot hold. See ADR 0015 for
  // why "always insert, then unconditionally trim to capacity" is
  // simpler and preferable here to a separate "reject oversized inserts"
  // or "exempt the just-inserted entry from eviction" special case.
  void Insert(std::uint64_t file_id, std::uint64_t block_offset, std::string block);

  // Drops every cached entry belonging to `file_id` -- for when a table
  // is no longer part of the DB (e.g. a future Phase 6 compaction
  // removing an obsolete table), so stale blocks can never be served
  // after the file itself is gone. Not exercised by anything in Phase
  // 4/5 (nothing removes a table yet); included now, on the cache's
  // public API, so Phase 6 does not need to modify this class.
  void Erase(std::uint64_t file_id);

  Stats GetStats() const;

 private:
  struct Key {
    std::uint64_t file_id = 0;
    std::uint64_t block_offset = 0;
    bool operator==(const Key& other) const {
      return file_id == other.file_id && block_offset == other.block_offset;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& key) const {
      // A simple, adequate combine for a two-field key -- this cache's
      // hash table only needs a reasonably uniform distribution across
      // buckets, not cryptographic or on-disk-format properties (unlike
      // bloom::BuildFilter's hash, which does need the latter).
      return std::hash<std::uint64_t>()(key.file_id) * 1099511628211ull ^
             std::hash<std::uint64_t>()(key.block_offset);
    }
  };

  using LruList = std::list<std::pair<Key, std::string>>;

  // Evicts least-recently-used entries (from the back of lru_) until
  // current_bytes_ <= capacity_bytes_. Caller must hold mutex_.
  void EvictToCapacityLocked();

  mutable std::mutex mutex_;
  std::size_t capacity_bytes_;
  std::size_t current_bytes_ = 0;
  // Most-recently-used at the front, least-recently-used at the back --
  // the classic intrusive-list LRU shape, letting Lookup()'s
  // promote-to-front and eviction's pop-from-back both be O(1).
  LruList lru_;
  std::unordered_map<Key, LruList::iterator, KeyHash> index_;
  Stats stats_;  // protected by mutex_, like everything else above
};

}  // namespace pebbledb::cache
