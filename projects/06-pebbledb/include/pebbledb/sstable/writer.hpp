#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pebbledb/entry_type.hpp"
#include "pebbledb/slice.hpp"
#include "pebbledb/sstable/format.hpp"
#include "pebbledb/status.hpp"
#include "pebbledb/util/posix_file.hpp"

namespace pebbledb::sstable {

// Writer builds one immutable, sorted SSTable file (spec roadmap Phase 3;
// FR3). Entries must be Add()ed in strictly increasing key order
// (invariant 3, "keys inside a sorted table obey comparator order") --
// Add() rejects an out-of-order or duplicate key with
// Status::InvalidArgument rather than silently producing an unsorted or
// ambiguous file. This matches the expected call pattern: flushing a
// MemTable's already-sorted entries() view (Phase 4) straight through,
// one Add() per map entry, in iteration order.
//
// Once Finish() returns OK, the file on disk is complete, fsynced, and
// never written to again by this Writer (invariant 2: "SSTable contents
// are immutable after successful creation") -- see docs/file-format.md
// and ADR 0009/ADR 0010.
class Writer {
 public:
  struct Options {
    // Target (not hard-cap) size, in bytes of encoded entries, of each
    // data block before it is closed and a new one started -- spec
    // decision D5's default (ADR 0009): 4 KiB. A single entry larger than
    // this target is still written whole into its own block rather than
    // being split, so an individual block's on-disk size can exceed this
    // target.
    std::size_t target_block_size_bytes = 4096;

    // Target bits per key for this table's Bloom filter block (spec FR3,
    // roadmap Phase 5; ADR 0014). 0 disables filter generation entirely
    // -- Finish() then emits the same zero-length, reserved filter block
    // every pre-Phase-5 table already had (ADR 0009). Default 10 matches
    // bloom::FilterPolicy's own default (~1% false-positive rate).
    std::size_t filter_bits_per_key = 10;
  };

  // Creates the SSTable file at `path` for writing. `path` must not
  // already have real content -- Open() uses util::PosixFile::OpenNew,
  // which fails with Status::IOError if `path` already has non-zero-byte
  // content, rather than appending to or truncating it (an absent or
  // already-empty path is fine and always starts a fresh file at offset
  // 0). So a caller that (by bug or a reused/retried file number) points
  // Open() at a path some other, real table already occupies gets an
  // error instead of a silently mis-offset, corrupted table (see ADR
  // 0011). Exactly like a brand-new WAL file (docs/durability.md),
  // creating the path also fsyncs the containing directory before this
  // returns OK.
  static Status Open(const std::string& path, const Options& options,
                      std::unique_ptr<Writer>* out);

  ~Writer() = default;
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  // Appends one entry. `key` must compare strictly greater than the key
  // passed to the previous Add() call (the first Add() has no such
  // constraint). Returns Status::InvalidArgument, without changing any
  // state that would appear in the eventual file, if this ordering
  // requirement is violated, or if called after Finish(). For
  // `type == EntryType::kTombstone`, `value` is ignored (encoded as
  // zero-length), exactly like a WAL kDelete record ignores its value.
  Status Add(Slice key, Slice value, std::uint64_t sequence, EntryType type);

  // Flushes any partially-filled data block, builds and writes the
  // filter block (real Bloom-filter content when
  // options.filter_bits_per_key > 0 and at least one entry was added;
  // the same zero-length, reserved region as every pre-Phase-5 table
  // otherwise -- ADR 0009/ADR 0014), the index block, and the footer,
  // then fsyncs the file. Add() must not be called after Finish()
  // returns (whether it returned OK or an error); Finish() itself must
  // not be called more than once. Calling either after a successful
  // Finish() returns Status::InvalidArgument without touching the file
  // again.
  Status Finish();

  std::uint64_t entries_written() const { return entries_written_; }

 private:
  Writer(std::unique_ptr<util::PosixFile> file, Options options)
      : file_(std::move(file)), options_(options) {}

  Status FlushCurrentBlock();

  std::unique_ptr<util::PosixFile> file_;
  Options options_;

  std::string current_block_entries_;  // encoded entries, not yet trailer-wrapped
  std::string last_key_in_current_block_;
  bool current_block_has_entries_ = false;

  std::string index_block_;        // accumulated encoded IndexEntry bytes
  std::uint64_t file_offset_ = 0;  // bytes appended so far (append-only; tracked ourselves)

  // Every key Add()ed so far, in order -- input to bloom::BuildFilter()
  // at Finish() (ADR 0014). Only populated when
  // options_.filter_bits_per_key > 0, to avoid the memory cost when
  // filter generation is disabled.
  std::vector<std::string> keys_for_filter_;

  std::string last_key_added_;
  bool has_added_any_ = false;
  std::uint64_t entries_written_ = 0;
  bool finished_ = false;
};

}  // namespace pebbledb::sstable
