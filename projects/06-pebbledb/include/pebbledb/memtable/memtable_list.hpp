#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>

#include "pebbledb/lookup_result.hpp"
#include "pebbledb/memtable/memtable.hpp"
#include "pebbledb/slice.hpp"

namespace pebbledb::memtable {

// MemTableList owns the "active MemTable + a newest-first list of frozen
// immutable MemTables" stack from the write/read path diagram
// (docs/architecture.md) and implements the read-side merge across it --
// spec roadmap Phase 2 exit criterion: "Reads merge active state
// correctly; flush can snapshot immutable state."
//
// - All writes (Put/Delete) go to the single active MemTable.
// - Freeze() moves the current active MemTable to the front of the
//   immutable list and replaces it with a brand-new, empty active
//   MemTable. The frozen MemTable is never written to again after this
//   point -- it is exactly the "immutable MemTable" stage of the roadmap
//   -- so a flush routine (Phase 3/4) can read its entries() at leisure,
//   independent of whatever the (new) active MemTable does afterward.
// - Get(key) checks the active MemTable first, then each immutable
//   MemTable from newest (front of the deque) to oldest (back), stopping
//   at the first layer that reports anything other than kNotFound. This
//   is the "newest layer wins outright" merge rule from
//   docs/architecture.md's read-path diagram: a tombstone in a newer
//   layer must never be shadowed by a stale value from an older layer
//   beneath it (invariant 4). There is no need to compare raw sequence
//   numbers *across* layers to get this right, because each individual
//   MemTable has already collapsed itself down to one, most-recent entry
//   per key (see memtable.hpp) -- layer order alone determines the
//   winner.
//
// Concurrency: MemTableList holds no lock of its own -- like MemTable, it
// expects its owner (DB, once wired in at Phase 4) to apply ADR 0003's
// single-writer/multi-reader discipline around Put/Delete/Freeze vs.
// Get. This class is a data structure, not a synchronization primitive.
class MemTableList {
 public:
  MemTableList();
  MemTableList(const MemTableList&) = delete;
  MemTableList& operator=(const MemTableList&) = delete;

  void Put(Slice key, Slice value, std::uint64_t sequence);
  void Delete(Slice key, std::uint64_t sequence);

  // Merges across the active MemTable and every immutable MemTable,
  // newest first -- see class comment.
  LookupResult Get(Slice key, std::string* value, std::uint64_t* sequence = nullptr) const;

  // Freezes the current active MemTable (pushed to the front of the
  // immutable list) and installs a fresh, empty active MemTable in its
  // place. Returns a shared_ptr to the now-immutable table so callers
  // (Phase 3/4 flush code) can keep reading entries() from it
  // independently of this MemTableList's subsequent lifetime and of any
  // further Put/Delete/Freeze calls made afterward.
  std::shared_ptr<MemTable> Freeze();

  // Removes `table` from the immutable list -- used once its data has
  // been durably captured elsewhere (Phase 4: flushed into an SSTable and
  // recorded in the manifest) and no longer needs to be kept resident in
  // RAM. A no-op if `table` is not currently present (e.g. it was already
  // removed, or points at some other MemTable entirely); this is
  // deliberately tolerant rather than asserting -- double-removal is
  // harmless, not a bug worth crashing a synchronous, single-writer flush
  // path over.
  void RemoveImmutable(const std::shared_ptr<MemTable>& table);

  const MemTable& active() const { return *active_; }
  std::size_t ActiveApproximateMemoryUsage() const { return active_->ApproximateMemoryUsage(); }
  bool ActiveShouldFlush(std::size_t size_threshold_bytes) const {
    return active_->ApproximateMemoryUsage() >= size_threshold_bytes;
  }

  std::size_t ImmutableCount() const { return immutables_.size(); }
  // Newest-first (front = most recently frozen), matching the merge
  // order Get() applies.
  const std::deque<std::shared_ptr<MemTable>>& immutables() const { return immutables_; }

 private:
  std::shared_ptr<MemTable> active_;
  std::deque<std::shared_ptr<MemTable>> immutables_;
};

}  // namespace pebbledb::memtable
