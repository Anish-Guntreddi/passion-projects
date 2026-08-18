#pragma once

#include <cstdint>
#include <string>

namespace pebbledb::util {

// DB directory file-naming scheme (spec roadmap Phase 4; ADR 0013). A
// persistent DB directory holds a fixed-name manifest
// (manifest::kManifestFileName -- see manifest/format.hpp) plus any
// number of *numbered* files: WAL segments and SSTables, each named from
// a single, monotonically increasing 64-bit file number
// (DB::next_file_number_, itself persisted as
// manifest::ManifestData::next_file_number so numbers are never reused
// across a restart). A file number is zero-padded to 6 decimal digits
// with a format-identifying extension -- e.g. file number 7 is
// "000007.wal" or "000007.sst". Six digits is generous headroom (up to
// 999,999 files) for this project's test/demo scale; a real long-running
// deployment would need a wider field or a different scheme, which is
// out of scope here. `file_number` is never reused for two different
// files/purposes -- a WAL file and an SSTable file never share the same
// number, since both are allocated from the one shared counter.
std::string WalFileName(const std::string& dbname, std::uint64_t file_number);
std::string SstableFileName(const std::string& dbname, std::uint64_t file_number);

}  // namespace pebbledb::util
