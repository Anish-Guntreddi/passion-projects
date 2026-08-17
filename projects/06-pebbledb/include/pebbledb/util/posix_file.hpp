#pragma once

#include <cstddef>
#include <cstdint>
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

  // Opens `path` for writing a brand-new file from scratch, failing with
  // Status::IOError if `path` already has real (non-zero-byte) content
  // rather than silently writing into/past it. A path that is absent, or
  // present but empty (0 bytes — e.g. a race-reserved placeholder), is
  // accepted identically either way and always yields a file starting at
  // offset 0. Use this -- never OpenAppend() -- for any writer whose
  // in-memory offset bookkeeping (e.g. sstable::Writer's file_offset_,
  // which starts at 0 and must exactly match on-disk byte positions)
  // would silently go wrong if `path` already had bytes in it.
  // OpenAppend() is deliberately reuse/resume-friendly (see its own
  // comment, and the WAL's "reopen for continued appending on restart"
  // use); that is the wrong primitive for a writer that must never be
  // pointed at a path with existing content. See ADR 0011.
  //
  // If this call creates `path`'s directory entry (it did not already
  // exist), the containing directory is fsync'd before this returns OK —
  // see "File creation durability: directory entries" in
  // docs/durability.md — mirroring OpenAppend()'s own rule.
  static Status OpenNew(const std::string& path, std::unique_ptr<PosixFile>* out);

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

  // Attempts to read exactly `n` bytes starting at absolute file offset
  // `offset`, via pread() -- this neither uses nor moves the sequential
  // position Read()/Append() otherwise track implicitly, so ReadAt() may
  // be freely interleaved with Read()/Append() calls on the same
  // PosixFile without disturbing either. Same short-read convention as
  // Read(): fewer than `n` bytes available before EOF is reported via
  // `*short_read`, not as an error. This is the random-access primitive
  // sstable::Reader uses for footer/index/data-block access (Phase 3) --
  // see ADR 0001's forward-looking note ("pread/pwrite semantics...
  // load-bearing once SSTables need random block access").
  Status ReadAt(std::uint64_t offset, std::size_t n, std::string* out, bool* short_read);

  // Returns the file's current size (via fstat()).
  Status Size(std::uint64_t* out);

 private:
  explicit PosixFile(int fd) : fd_(fd) {}

  int fd_ = -1;
};

}  // namespace pebbledb::util
