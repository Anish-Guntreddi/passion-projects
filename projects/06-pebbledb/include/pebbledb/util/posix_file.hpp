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

  // Opens `path` for writing, creating it if absent and truncating it to
  // zero length if it already has content -- unconditionally, with no
  // "must be absent/empty" check (contrast OpenNew()). This is the right
  // primitive for a disposable, reused-by-name scratch/staging file (e.g.
  // manifest::Save()'s "write-new + fsync + atomic rename" temp file,
  // ADR 0012/D7) that is always meant to be clobbered by the next writer,
  // as opposed to a file (like an SSTable) where silently overwriting
  // real prior content would itself be the bug OpenNew() exists to catch.
  //
  // If this call creates `path`'s directory entry (it did not already
  // exist), the containing directory is fsync'd before this returns OK --
  // same rule as OpenAppend()/OpenNew() (docs/durability.md, "File
  // creation durability: directory entries").
  static Status OpenTruncate(const std::string& path, std::unique_ptr<PosixFile>* out);

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

// Fsyncs the directory at `dir_path` directly. fsync() on a file
// descriptor only guarantees that *file's* contents are durable; it does
// not, on POSIX filesystems, guarantee that a directory entry pointing
// into that directory (a newly created file, or a name freshly
// rename()'d into place) survives a crash -- that requires this separate
// call. OpenAppend()/OpenNew()/OpenTruncate() already call this
// internally for the "this call created a new directory entry" case;
// it is exposed here as its own function because some callers need it
// for an operation this class has no dedicated method for -- e.g.
// manifest::Save()'s post-rename fsync (ADR 0012/D7): renaming
// MANIFEST.tmp over MANIFEST changes what the name "MANIFEST" points to,
// which is exactly the same kind of directory-entry mutation as a brand
// new file being created, and needs the same durability treatment. See
// docs/durability.md, "File creation durability: directory entries".
Status SyncDirectory(const std::string& dir_path);

// Truncates the file at `path` to exactly `length` bytes (POSIX
// semantics: shrinking discards trailing bytes; `length` must not exceed
// the file's current size -- this is a "discard untrusted trailing
// bytes" primitive, never a "grow/pad" one) and fsyncs the result before
// returning. This is a plain content mutation of an existing file, not a
// directory-entry creation/replacement, so unlike SyncDirectory() above
// there is no separate containing-directory fsync to perform.
//
// Used by DB::Recover() (see wal::Reader::valid_prefix_length()) to
// discard an untrusted torn trailing WAL record *before* that file is
// ever reopened for append: without this, a second unclean shutdown
// could leave new, valid records concatenated directly after the old
// torn bytes, which no later replay could safely tell apart from a
// single corrupted record (docs/durability.md, ADR 0016).
Status TruncateFile(const std::string& path, std::uint64_t length);

}  // namespace pebbledb::util
