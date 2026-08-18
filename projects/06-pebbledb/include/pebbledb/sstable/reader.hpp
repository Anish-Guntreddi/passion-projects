#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pebbledb/cache/block_cache.hpp"
#include "pebbledb/lookup_result.hpp"
#include "pebbledb/slice.hpp"
#include "pebbledb/sstable/format.hpp"
#include "pebbledb/status.hpp"
#include "pebbledb/util/posix_file.hpp"

namespace pebbledb::sstable {

// Reader opens an SSTable file written by Writer and serves point lookups
// (spec roadmap Phase 3 exit criterion: "Table written by writer reopens
// and queries after process restart") and full-table iteration (used by
// tools/inspect_sstable and, in later phases, by compaction).
//
// Open() reads and validates the footer, then fully loads the (small,
// sparse per FR3) index block into memory, and -- Phase 5 -- the filter
// block if one is present (footer.filter_handle.size > 0; ADR 0014) --
// all eagerly, so any footer/index/filter corruption is caught at Open()
// time rather than resurfacing unpredictably on some later Get() call.
// Data blocks are read lazily, on demand, per Get()/ForEachEntry() call,
// via random-access pread (util::PosixFile::ReadAt) -- see ADR 0001's
// note on pread becoming load-bearing at this phase, and ADR 0010 -- or
// served from an optional shared cache::BlockCache (Phase 5, spec FR7;
// ADR 0015) when one is supplied to Open().
//
// Corruption handling (invariant 7): unlike wal::Reader, which is a
// single sequential cursor that becomes permanently exhausted at the
// first bad record, sstable::Reader has no such single cursor -- each
// Get()/ForEachEntry() call independently reads only the block(s) it
// needs. A corrupted data block therefore only fails the lookups that
// happen to land in that specific block; lookups resolved by other,
// uncorrupted blocks are unaffected. Any corruption that IS encountered
// is always reported as a non-OK Status, never silently folded into
// kNotFound -- see Get()'s contract below.
class Reader {
 public:
  // `cache`, if non-null, is a block cache shared across every open
  // Reader in the owning DB (ADR 0015) -- not owned by this Reader, and
  // must outlive it. `file_id` is the cache key's file-identifying half
  // (D8's default: (file_id, block_offset) -- see cache/block_cache.hpp)
  // and should be unique per on-disk table within whatever scope `cache`
  // is shared over (DB uses the table's file number -- ADR 0013). Both
  // parameters are ignored (no caching happens) when `cache == nullptr`,
  // which is also what every pre-Phase-5 call site (existing tests,
  // tools/inspect_sstable) gets via the defaults.
  static Status Open(const std::string& path, std::unique_ptr<Reader>* out,
                      cache::BlockCache* cache = nullptr, std::uint64_t file_id = 0);

  ~Reader() = default;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  // Looks up `key`. On Status::OK, `*result` reports:
  //  - kFound: `*value` (and, if non-null, `*sequence`) are populated.
  //  - kDeleted: this table holds a tombstone for `key`; `*value` is left
  //    untouched.
  //  - kNotFound: `key` never appeared in this table -- including when a
  //    loaded Bloom filter (Phase 5) reports `key` definitely absent,
  //    which short-circuits this call before any data-block read
  //    happens (filter_checks()/filter_rejections() count exactly this).
  // A non-OK Status (Corruption/IOError/NotSupported) means the lookup
  // could not be completed at all -- e.g. a block's checksum failed to
  // verify -- and `*result` is left unspecified; invariant 7 requires
  // this be surfaced as an error, never silently folded into kNotFound.
  Status Get(Slice key, LookupResult* result, std::string* value,
             std::uint64_t* sequence = nullptr) const;

  using EntryCallback = std::function<void(const Entry& entry)>;

  // Invokes `callback` once per entry, in ascending key order, reading
  // every data block sequentially (through the cache, if one was
  // supplied to Open() -- same as Get()). Used by tools/inspect_sstable
  // and intended for future compaction (Phase 6). Stops and returns a
  // non-OK Status on the first I/O/corruption error encountered.
  Status ForEachEntry(const EntryCallback& callback) const;

  std::uint64_t num_entries() const { return footer_.num_entries; }
  std::size_t index_entry_count() const { return index_.size(); }
  const Footer& footer() const { return footer_; }
  // Ascending by last_key -- the on-disk index order.
  const std::vector<IndexEntry>& index_entries() const { return index_; }

  // True if this table has a loaded Bloom filter (footer.filter_handle
  // was non-empty at Open() time) -- Phase 5, ADR 0014.
  bool has_filter() const { return !filter_.empty(); }
  // Every Get() call that consulted the filter, and how many of those
  // were short-circuited by a definite-absence answer (i.e. avoided a
  // data-block read entirely) -- exposed so a benchmark/test can
  // directly measure the "reduced unnecessary reads" roadmap Phase 5
  // exit criterion, rather than inferring it indirectly. Both 0 if
  // has_filter() is false. Atomic: Get() is const and safe to call
  // concurrently (multiple DB::Get() calls only take DB's shared lock --
  // ADR 0003), so these counters must be too.
  std::uint64_t filter_checks() const { return filter_checks_.load(std::memory_order_relaxed); }
  std::uint64_t filter_rejections() const {
    return filter_rejections_.load(std::memory_order_relaxed);
  }
  // Data-block cache hits/misses attributable to this Reader (0/0 if no
  // cache was supplied to Open()) -- see cache_hits()/cache_misses() and
  // ADR 0015.
  std::uint64_t cache_hits() const { return cache_hits_.load(std::memory_order_relaxed); }
  std::uint64_t cache_misses() const { return cache_misses_.load(std::memory_order_relaxed); }
  // Number of times this Reader actually issued a pread() for a *data*
  // block (i.e. neither a filter rejection nor a cache hit satisfied the
  // request) -- independent of whether a cache is in use at all, so a
  // test/benchmark can directly measure "reads avoided" by the filter
  // and/or the cache rather than inferring it indirectly. See
  // tests/integration/test_sstable_filter_and_cache.cpp.
  std::uint64_t data_blocks_read() const { return data_blocks_read_.load(std::memory_order_relaxed); }

 private:
  explicit Reader(std::unique_ptr<util::PosixFile> file) : file_(std::move(file)) {}

  Status ReadBlock(const BlockHandle& handle, std::string* raw_block,
                    std::string_view* entries) const;

  // Reads a *data* block's entries, trailer already verified and
  // stripped, into `*entries_owned` -- via the shared cache_ (if any),
  // falling back to ReadBlock() on a cache miss and populating the cache
  // for next time. Used by Get()/ForEachEntry() instead of calling
  // ReadBlock() directly, so both go through the same cache path (ADR
  // 0015). Index/filter blocks are never cached (see class comment) and
  // still go through plain ReadBlock(), at Open() only.
  Status ReadDataBlock(const BlockHandle& handle, std::string* entries_owned) const;

  std::unique_ptr<util::PosixFile> file_;
  Footer footer_;
  std::vector<IndexEntry> index_;
  // Owned, trailer-stripped filter-block content (bloom::BuildFilter's
  // output format -- see bloom/bloom_filter.hpp); empty if
  // footer_.filter_handle.size == 0 (no filter present -- ADR 0009).
  std::string filter_;
  // The file's size as of Open() (files are never written to again after
  // Open() -- invariant 2), used to bounds-check every BlockHandle
  // (footer's index_handle/filter_handle, and each index entry's data
  // block handle) before it ever drives a pread/allocation -- see
  // ReadBlock() and ADR 0011.
  std::uint64_t file_size_ = 0;

  cache::BlockCache* cache_ = nullptr;  // not owned; may be nullptr -- see Open()
  std::uint64_t file_id_ = 0;
  mutable std::atomic<std::uint64_t> filter_checks_{0};
  mutable std::atomic<std::uint64_t> filter_rejections_{0};
  mutable std::atomic<std::uint64_t> cache_hits_{0};
  mutable std::atomic<std::uint64_t> cache_misses_{0};
  mutable std::atomic<std::uint64_t> data_blocks_read_{0};
};

}  // namespace pebbledb::sstable
