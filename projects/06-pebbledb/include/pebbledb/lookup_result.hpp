#pragma once

namespace pebbledb {

// LookupResult is the three-state answer a single LSM layer (a MemTable,
// a MemTableList, or an SSTable Reader) gives for a point lookup:
//
//  - kFound:    this layer holds a live value for the key.
//  - kDeleted:  this layer holds a tombstone for the key (spec FR4). The
//               caller must stop searching older/lower layers -- treating
//               kDeleted the same as kNotFound and falling through would
//               let a stale value from an older layer resurface, which is
//               exactly what invariant 4 (spec §1.5) prohibits.
//  - kNotFound: this layer has no record of the key at all (neither a
//               value nor a tombstone). The caller may continue searching
//               older/lower layers.
//
// This three-state distinction is why every layer's lookup API returns
// LookupResult instead of collapsing "deleted" into "not found" the way
// DB::Get's Phase-0 two-state Status (OK vs. NotFound) does -- DB::Get
// only needs to make that collapse once, at the top of the whole merged
// read path (Phase 4 onward), not at every intermediate layer.
enum class LookupResult {
  kFound,
  kDeleted,
  kNotFound,
};

const char* LookupResultName(LookupResult result);

}  // namespace pebbledb
