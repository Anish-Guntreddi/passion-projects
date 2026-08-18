#pragma once

#include <string>

#include "pebbledb/manifest/format.hpp"
#include "pebbledb/status.hpp"

namespace pebbledb::manifest {

// I/O layer over the manifest v1 binary format (format.hpp) -- the
// manifest counterpart of wal::Writer/Reader and sstable::Writer/Reader
// owning the file I/O side of their respective formats, while format.hpp
// stays pure encode/decode. See ADR 0012 (binary format, D7's atomicity
// approach) and ADR 0013 (how DB::Recover()/DB::Flush() use these two
// functions).

// Loads and decodes the manifest file at `dbname`/kManifestFileName.
// Returns:
//  - OK, `*out` populated -- a valid manifest was found and decoded.
//  - NotFound -- no manifest file exists yet at this path (a brand-new,
//    empty DB directory). Not an error: DB::Recover() (ADR 0013) treats
//    this as "start fresh, allocate everything from scratch."
//  - Corruption -- a manifest file exists but failed to decode (bad
//    magic/version/length/checksum -- format::DecodeStatus, named via
//    DecodeStatusName, folded into the message). Per invariant 5 ("the
//    manifest never intentionally references a partially published
//    table") this should never happen from this project's own Save()
//    calls (write-new + fsync + atomic rename means a reader only ever
//    observes a fully-old or fully-new manifest, never a torn one) --
//    seeing it means something external corrupted the file, which this
//    project deliberately does not attempt to auto-recover from (fails
//    DB::Open() outright rather than silently discarding the table list
//    and losing track of what is safe to read).
//  - IOError -- some other I/O failure while reading the file.
Status LoadManifest(const std::string& dbname, ManifestData* out);

// Atomically replaces the on-disk manifest for `dbname` with `data`'s
// encoding: write-new + fsync + atomic rename (spec decision D7; ADR
// 0012).
//
//  1. Encode `data` and write it to `dbname`/kManifestTempFileName,
//     truncating any stale leftover temp file from a prior crashed
//     Save() call (util::PosixFile::OpenTruncate -- a previous crash
//     mid-Save() can only have left an incomplete *temp* file; the real
//     manifest at kManifestFileName is untouched until the rename below,
//     so there is nothing to recover from an abandoned temp file except
//     to discard it).
//  2. fsync that temp file, so its bytes are durable before anything
//     references it by its final name.
//  3. rename() the temp file over `dbname`/kManifestFileName -- atomic on
//     the same filesystem (POSIX guarantees a concurrent reader/opener
//     of that path sees either the old or the new complete file, never a
//     mix) -- this is the actual publish point.
//  4. fsync the containing directory, so the rename's directory-entry
//     mutation is itself durable (docs/durability.md's "File creation
//     durability: directory entries" rule, generalized from file
//     *creation* to file *replacement* -- both mutate what a directory
//     entry points to, and both need this same treatment).
//
// `*published` (if non-null) is always set before returning, and
// distinguishes two very different failure shapes a caller must not
// treat identically (ADR 0016):
//  - false: step 3's rename() never happened (steps 1-2 failed, or the
//    rename() call itself failed). `dbname`'s on-disk manifest is
//    *unchanged* from whatever it was before this call -- a caller may
//    safely treat this exactly like "this call had no effect at all" and
//    retry later from the same state.
//  - true: rename() succeeded -- `dbname`/kManifestFileName **on disk
//    already is** `data`'s encoding -- but step 4's directory fsync
//    failed, so that rename's own durability across a crash is not
//    guaranteed. The manifest **is published**; a caller must not
//    continue operating as if this call had no effect (doing so is
//    exactly how a live process's in-memory state can silently diverge
//    from what a subsequent crash's `LoadManifest()` will actually see —
//    see DB::FlushLocked(), which adopts `data` into its in-memory state
//    in this case rather than continuing to serve out of whichever state
//    preceded this call).
Status SaveManifest(const std::string& dbname, const ManifestData& data, bool* published = nullptr);

}  // namespace pebbledb::manifest
