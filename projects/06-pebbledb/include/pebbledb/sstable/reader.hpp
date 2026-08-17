#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
// sparse per FR3) index block into memory -- both eagerly, so any
// footer/index corruption is caught at Open() time rather than
// resurfacing unpredictably on some later Get() call. Data blocks are
// read lazily, on demand, per Get()/ForEachEntry() call, via
// random-access pread (util::PosixFile::ReadAt) -- see ADR 0001's note on
// pread becoming load-bearing at this phase, and ADR 0010.
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
  static Status Open(const std::string& path, std::unique_ptr<Reader>* out);

  ~Reader() = default;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  // Looks up `key`. On Status::OK, `*result` reports:
  //  - kFound: `*value` (and, if non-null, `*sequence`) are populated.
  //  - kDeleted: this table holds a tombstone for `key`; `*value` is left
  //    untouched.
  //  - kNotFound: `key` never appeared in this table.
  // A non-OK Status (Corruption/IOError/NotSupported) means the lookup
  // could not be completed at all -- e.g. a block's checksum failed to
  // verify -- and `*result` is left unspecified; invariant 7 requires
  // this be surfaced as an error, never silently folded into kNotFound.
  Status Get(Slice key, LookupResult* result, std::string* value,
             std::uint64_t* sequence = nullptr) const;

  using EntryCallback = std::function<void(const Entry& entry)>;

  // Invokes `callback` once per entry, in ascending key order, reading
  // every data block sequentially. Used by tools/inspect_sstable and
  // intended for future compaction (Phase 6). Stops and returns a
  // non-OK Status on the first I/O/corruption error encountered.
  Status ForEachEntry(const EntryCallback& callback) const;

  std::uint64_t num_entries() const { return footer_.num_entries; }
  std::size_t index_entry_count() const { return index_.size(); }
  const Footer& footer() const { return footer_; }
  // Ascending by last_key -- the on-disk index order.
  const std::vector<IndexEntry>& index_entries() const { return index_; }

 private:
  explicit Reader(std::unique_ptr<util::PosixFile> file) : file_(std::move(file)) {}

  Status ReadBlock(const BlockHandle& handle, std::string* raw_block,
                    std::string_view* entries) const;

  std::unique_ptr<util::PosixFile> file_;
  Footer footer_;
  std::vector<IndexEntry> index_;
  // The file's size as of Open() (files are never written to again after
  // Open() -- invariant 2), used to bounds-check every BlockHandle
  // (footer's index_handle/filter_handle, and each index entry's data
  // block handle) before it ever drives a pread/allocation -- see
  // ReadBlock() and ADR 0011.
  std::uint64_t file_size_ = 0;
};

}  // namespace pebbledb::sstable
