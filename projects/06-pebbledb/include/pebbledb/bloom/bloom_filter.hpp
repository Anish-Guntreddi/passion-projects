#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pebbledb::bloom {

// Bloom filter build/query for SSTable Phase 5 filter blocks (spec FR3,
// "per-table (or block-group) Bloom filter"; roadmap Phase 5 exit
// criterion, "negative-read filtering"). One filter per table, covering
// every key written to it -- spec FR3 explicitly permits either
// per-table or per-block-group; per-table is simpler and sufficient at
// this project's scale (ADR 0014).
//
// On-disk content format (the bytes that end up inside the SSTable's
// filter block -- see sstable/format.hpp's BlockHandle/filter_handle and
// docs/file-format.md; this content is wrapped by the same
// FinishBlock()/VerifyAndStripBlockTrailer() convention as every other
// block, so it gets its own CRC-32C checksum for free, exactly like a
// data or index block):
//
//   num_probes    uint8    number of hash probes (k) used to build this
//                          filter -- stored explicitly so a reader never
//                          has to assume a particular bits-per-key/k
//                          configuration; self-describing, matching this
//                          project's "structural fields before content"
//                          convention elsewhere (ADR 0014).
//   bits          bytes    the bit array itself, exactly
//                          (content.size() - 1) bytes.
//
// False positives are possible ("may be present" for a key never
// added); false negatives are not ("definitely absent" is only ever
// returned for a key that truly was never added). A false positive costs
// an unnecessary data-block read; a false negative would be a
// correctness bug (silently hiding a real key) -- sstable::Reader::Get()
// only ever uses a filter's negative answer to *skip* a block read it
// would otherwise have had to do anyway, never to skip validating
// anything else.
struct FilterPolicy {
  // Target bits per key (ADR 0014's default: 10, matching the widely
  // used LevelDB/RocksDB default -- see that ADR for the
  // false-positive-rate math). 0 disables filter generation entirely:
  // BuildFilter() then returns an empty string, and
  // sstable::Writer::Finish() emits the same zero-length, reserved
  // filter block every pre-Phase-5 table already had (ADR 0009).
  std::size_t bits_per_key = 10;
};

// Builds a filter over `keys` per `policy`. Returns the encoded content
// described above, ready to be wrapped by sstable::FinishBlock(). Returns
// an empty string if `keys` is empty or `policy.bits_per_key == 0` --
// callers must treat an empty result as "no filter", exactly like the
// SSTable footer's `filter_handle.size == 0` convention (ADR 0009).
std::string BuildFilter(const std::vector<std::string>& keys, const FilterPolicy& policy);

// Returns true if `key` *may* be present in a table whose filter-block
// content (the format above) is `filter` -- a false positive is
// possible, a false negative is not (see the struct comment above).
// Returns true (conservatively "may match", i.e. "do the real lookup")
// if `filter` is empty or too short to parse as a valid filter, rather
// than ever risking treating a malformed/corrupted filter as grounds to
// report a real key absent -- a corrupt or missing filter degrades to
// "no filtering happened", never to "silently hide real keys" (see ADR
// 0014).
bool KeyMayMatch(std::string_view key, std::string_view filter);

}  // namespace pebbledb::bloom
