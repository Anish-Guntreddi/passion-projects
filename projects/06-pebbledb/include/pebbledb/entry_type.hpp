#pragma once

#include <cstdint>

namespace pebbledb {

// EntryType distinguishes a live value from a tombstone (spec FR4,
// invariant 4) for any layer that stores the LSM's "current resolved
// state" per key: MemTable (Phase 2, include/pebbledb/memtable/) and
// SSTable (Phase 3, include/pebbledb/sstable/). The two layers share one
// enum, rather than each defining its own, because a flush (Phase 4)
// passes an entry from a MemTable straight through into an SSTable with
// no semantic transformation -- it is the same concept at both layers.
//
// wal::RecordType (wal/record.hpp) is deliberately a *separate*,
// WAL-specific type: a WAL record represents "an operation that was
// requested" (an event in a durability log), not "the LSM's current
// resolved state for this key" (a fact about what a MemTable/SSTable
// currently holds). Code that replays a WAL record into a MemTable
// (Phase 4) maps wal::RecordType -> EntryType explicitly at that one call
// site; the two vocabularies are allowed to diverge without forcing this
// header to change.
enum class EntryType : std::uint8_t {
  kValue = 1,
  kTombstone = 2,
};

const char* EntryTypeName(EntryType type);

}  // namespace pebbledb
