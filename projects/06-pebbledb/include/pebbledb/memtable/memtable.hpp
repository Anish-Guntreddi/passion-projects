#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "pebbledb/entry_type.hpp"
#include "pebbledb/lookup_result.hpp"
#include "pebbledb/slice.hpp"

namespace pebbledb::memtable {

// MemTable: the LSM's in-memory ordered write buffer (spec roadmap Phase
// 2; FR2). Backed by std::map -- spec decision D4's default (ADR 0008); a
// skip list is an explicitly out-of-scope stretch goal (spec Part 3).
//
// Semantics:
//  - Ordered by key (comparator order -- this is what makes an immutable
//    MemTable's entries() the right shape to feed straight into a Phase 3
//    SSTable writer, which requires strictly increasing key order).
//  - Each key holds exactly one entry: the most recent Put/Delete for that
//    key overwrites whatever was there before, carrying forward the
//    sequence number and type (kValue/kTombstone) of that most recent
//    write (spec decision D2 / ADR 0005: the sequence number is the
//    cross-write ordering key). This mirrors DB's Phase 0 std::map
//    behavior with two additions: an explicit tombstone marker instead of
//    physically erasing the key (see "why tombstones, not erasure" below)
//    and a sequence number carried per entry.
//  - MemTable does not itself allocate sequence numbers; the caller
//    supplies one per Put/Delete call (mirrors ADR 0005's "Scope for
//    Phase 1" note: a single global counter lives with DB once the WAL
//    and MemTable are wired into the write path, which is Phase 4).
//
// Why tombstones, not erasure: a MemTable is one layer in a stack (active
// -> immutable MemTable(s) -> SSTables, newest first --
// docs/architecture.md's read-path diagram). A Delete must be visible to
// that merge so a lookup that reaches this layer and finds a tombstone
// stops right there and reports "not found", instead of falling through
// to a stale value in an older immutable MemTable or SSTable underneath
// it (invariant 4, spec §1.5). Physically erasing the key here would make
// a delete indistinguishable from "this layer never heard of the key",
// which is exactly the bug invariant 4 rules out -- see
// memtable_list.hpp for where this matters most (merging across layers).
//
// Concurrency: MemTable itself holds no lock. Like the rest of this
// phase's code, it expects its owner (MemTableList, and eventually DB at
// Phase 4) to apply ADR 0003's single-writer/multi-reader discipline
// around Put/Delete vs. Get/entries(). MemTable is a data structure, not
// a synchronization primitive.
class MemTable {
 public:
  struct Entry {
    std::uint64_t sequence = 0;
    EntryType type = EntryType::kValue;
    std::string value;  // empty and meaningless for kTombstone
  };

  // Default size threshold (spec FR2, "size threshold"). See ADR 0008 for
  // why 4 MiB and why this is an approximate, not exact, accounting.
  static constexpr std::size_t kDefaultSizeThresholdBytes = 4u * 1024 * 1024;

  MemTable() = default;
  MemTable(const MemTable&) = delete;
  MemTable& operator=(const MemTable&) = delete;

  // Inserts or overwrites `key` with a live value at `sequence`. `key`/
  // `value` are copied into MemTable-owned storage before this returns
  // (slice.hpp ownership rules, ADR 0002). No failure mode exists at this
  // layer (barring allocation failure, which this codebase does not
  // handle anywhere else either -- see DB::Put) -- hence `void`, not
  // `Status`.
  void Put(Slice key, Slice value, std::uint64_t sequence);

  // Records a tombstone for `key` at `sequence` -- see class comment,
  // "why tombstones, not erasure". Idempotent and unconditional: deleting
  // an absent or already-deleted key still (re)writes the tombstone at
  // the new sequence, exactly as a Put would overwrite a prior value.
  void Delete(Slice key, std::uint64_t sequence);

  // Looks up `key` in this MemTable alone -- no merging with any other
  // layer (see MemTableList::Get for that). Returns:
  //  - kFound: `*value` holds the live value, and `*sequence` (if
  //    non-null) the entry's sequence number.
  //  - kDeleted: this MemTable holds a tombstone for `key`. `*value` is
  //    left untouched; `*sequence` (if non-null) is still populated with
  //    the tombstone's sequence number.
  //  - kNotFound: this MemTable has never recorded anything for `key`.
  LookupResult Get(Slice key, std::string* value, std::uint64_t* sequence = nullptr) const;

  // Ascending-key-order view of every entry currently in this MemTable --
  // exactly the order a Phase 3 SSTable writer requires (invariant 3:
  // "keys inside a sorted table obey comparator order"). Read-only; the
  // caller must not mutate this MemTable while iterating the returned
  // reference (this project's design has no writer ever touch a MemTable
  // again once it is frozen -- see memtable_list.hpp's Freeze() -- so a
  // flush routine iterating a frozen table's entries() is always safe).
  const std::map<std::string, Entry, std::less<>>& entries() const { return table_; }

  std::size_t entry_count() const { return table_.size(); }
  bool empty() const { return table_.empty(); }

  // Sum, over every entry currently in this MemTable, of key size + value
  // size (0 for a tombstone) + kPerEntryOverheadBytes. This is an
  // approximation of actual memory use (spec FR2 just says "size
  // threshold", not an exact-byte accounting), tracked incrementally so
  // it stays O(1) per Put/Delete rather than re-summing the whole map --
  // see ADR 0008.
  std::size_t ApproximateMemoryUsage() const { return approximate_memory_usage_; }

  // A rough stand-in for std::map's actual per-node overhead (red-black
  // tree node header + allocator bookkeeping), which varies by
  // standard-library implementation and is deliberately not modeled
  // exactly -- see ADR 0008. Public so memtable.cpp's free-function cost
  // helper can share this one definition rather than duplicating the
  // constant.
  static constexpr std::size_t kPerEntryOverheadBytes = 32;

 private:
  std::map<std::string, Entry, std::less<>> table_;
  std::size_t approximate_memory_usage_ = 0;
};

}  // namespace pebbledb::memtable
