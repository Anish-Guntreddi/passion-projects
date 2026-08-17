#pragma once

#include <cstdint>
#include <string>

namespace pebbledb::testutil {

// RAII wrapper for a unique, empty path under the OS temp directory,
// removed on destruction. Reserves the name race-free (via mkstemp)
// without creating any content — tests create the actual file content
// through whatever API is under test (wal::Writer::Open, etc.).
class TempFile {
 public:
  TempFile();
  ~TempFile();
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Returns the size in bytes of the file at `path`, or -1 if it could not
// be stat'd.
std::int64_t FileSize(const std::string& path);

// Returns a path, under the OS temp directory, that does not exist yet —
// unlike TempFile (which pre-creates an empty file via mkstemp so the
// reserved path already exists), this is for tests that need to exercise
// an API's "create a brand-new file" code path specifically (e.g.
// util::PosixFile::OpenAppend's directory-fsync-on-creation behavior —
// see docs/durability.md). Implemented via mkstemp-then-unlink, so the
// name is reserved race-free before being freed again; as with any
// "guaranteed unique, not guaranteed to stay free" name, a concurrent
// process could theoretically claim it before the caller does, which is
// an accepted, standard trade-off for this kind of test helper. Throws
// std::runtime_error on failure.
std::string UniqueNonExistentPath();

// Truncates/resizes the file at `path` to exactly `length` bytes (POSIX
// truncate semantics: shrinking discards trailing bytes). Used by
// recovery/corruption tests to simulate a torn write at an arbitrary
// byte offset. Throws std::runtime_error on failure.
void TruncateFileTo(const std::string& path, std::int64_t length);

// Reads the entire file at `path` into memory. Throws std::runtime_error
// on failure.
std::string ReadWholeFile(const std::string& path);

// Writes `contents` to `path`, truncating/creating it first (i.e. the
// file's final contents are exactly `contents`). Throws
// std::runtime_error on failure.
void WriteWholeFile(const std::string& path, const std::string& contents);

// XORs the single byte at `byte_offset` in the file at `path` with
// `mask` (default 0xFF: flips every bit in that byte). Used by
// corruption tests to simulate bit rot / a garbled write without
// changing the file's length. Throws std::runtime_error on failure
// (including if `byte_offset` is at or past the file's current length).
void CorruptByteAt(const std::string& path, std::int64_t byte_offset,
                    unsigned char mask = 0xFF);

}  // namespace pebbledb::testutil
