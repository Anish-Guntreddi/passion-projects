#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "pebbledb/status.hpp"

namespace pebbledb::util {

// RAII owner of a POSIX file descriptor used for WAL I/O.
//
// Access pattern (deliberately sequential, not positional): the WAL is
// append-only and is read back front-to-back during recovery, so plain
// write()/read()/fsync() on a single fd with an implicit, kernel-tracked
// file offset is sufficient and simpler than pread()/pwrite(). The tech
// stack plan's "portable pread/pwrite/fsync semantics first" guidance
// becomes load-bearing once SSTables need random block access (Phase 3);
// see docs/decisions/0001-build-tooling-and-warnings.md for the scoping
// note.
class PosixFile {
 public:
  // Opens `path` for appending, creating it (mode 0644) if it does not
  // already exist. The fd is opened with O_APPEND so every Append() call
  // is atomically positioned at the current end of file even if another
  // process/fd extended it concurrently (not a scenario PebbleDB's
  // single-writer model relies on today, but it is the correct primitive
  // regardless — see ADR 0003).
  //
  // If this call is the one that creates `path` (it did not already
  // exist), the containing directory is fsync'd before this returns OK —
  // fsync() on the file itself only makes its *contents* durable, not the
  // directory entry pointing to a brand-new file; see docs/durability.md,
  // "File creation durability: directory entries". Reopening an existing
  // path does not re-fsync the directory (no new entry was created).
  static Status OpenAppend(const std::string& path, std::unique_ptr<PosixFile>* out);

  // Opens `path` read-only. Fails with Status::IOError if the path does
  // not exist.
  static Status OpenRead(const std::string& path, std::unique_ptr<PosixFile>* out);

  ~PosixFile();
  PosixFile(const PosixFile&) = delete;
  PosixFile& operator=(const PosixFile&) = delete;

  // Appends `data` to the file. Does not fsync — durability is the
  // caller's responsibility via Sync(), per the WAL's SyncPolicy
  // (wal/writer.hpp).
  Status Append(std::string_view data);

  // Fsyncs all previously written bytes to stable storage.
  Status Sync();

  // Attempts to read exactly `n` bytes sequentially, starting at the
  // file's current position, into `*out` (which is overwritten). If
  // fewer than `n` bytes remain before EOF, `*out` contains whatever was
  // actually available (0..n-1 bytes) and `*short_read` is set to true;
  // this is how wal::Reader distinguishes a complete record from a torn
  // trailing write. Returns a non-OK Status only on an actual OS-level
  // I/O error (not on ordinary EOF, which is reported via `*short_read`).
  Status Read(std::size_t n, std::string* out, bool* short_read);

 private:
  explicit PosixFile(int fd) : fd_(fd) {}

  int fd_ = -1;
};

}  // namespace pebbledb::util
