#pragma once

#include <unistd.h>

#include <utility>

namespace arcserve::net {

// RAII owner of a single POSIX file descriptor.
//
// Invariants:
//   - At most one live FileDescriptor owns a given fd value at a time.
//     Ownership transfers only via move; copy is disabled so two objects
//     can never both believe they own (and may both close) the same fd.
//   - `-1` is the canonical "no fd owned" sentinel, matching POSIX's own
//     convention for "invalid descriptor". A default-constructed or
//     moved-from FileDescriptor holds -1 and closes nothing on destruction.
//   - The destructor closes the owned fd exactly once, if valid. Nothing
//     else in this codebase is permitted to call ::close() on a raw fd that
//     is (or was) owned by a FileDescriptor — that would violate "every fd
//     has an obvious owner" (spec core design principle).
//   - close() failures are swallowed in the destructor (there is no way to
//     surface an error from a destructor safely); explicit close() before
//     destruction is provided for callers who need to observe close(2)
//     failures such as EIO on the final flush of buffered writes.
//
// Not thread-safe: a single FileDescriptor instance must not be accessed
// from multiple threads without external synchronization. Distinct
// FileDescriptor instances (owning distinct fds) are independent.
class FileDescriptor {
 public:
  FileDescriptor() noexcept = default;

  // Takes ownership of `fd`. `fd` may be -1 (meaning "no descriptor"); any
  // other negative value is a caller error but is still accepted as-is
  // (this constructor performs no syscalls and cannot fail).
  explicit FileDescriptor(int fd) noexcept : fd_(fd) {}

  ~FileDescriptor() { reset(); }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      reset(std::exchange(other.fd_, -1));
    }
    return *this;
  }

  // True if this object currently owns an fd (i.e. holds a value != -1).
  // Does NOT verify the fd is still open at the OS level.
  [[nodiscard]] bool valid() const noexcept { return fd_ != -1; }
  explicit operator bool() const noexcept { return valid(); }

  // Borrow the underlying fd for use in a syscall. The returned value is
  // only valid for as long as this FileDescriptor is alive and untouched.
  [[nodiscard]] int get() const noexcept { return fd_; }

  // Give up ownership without closing. The caller becomes responsible for
  // the fd's lifetime. Returns -1 if this object owned nothing.
  [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

  // Close the currently-owned fd (if any) and take ownership of `new_fd`
  // (default -1, i.e. "own nothing"). Returns the close(2) result for the
  // previously-owned fd: 0 on success, -1 on failure (errno set), or 0 if
  // nothing was owned. Safe to call redundantly.
  int reset(int new_fd = -1) noexcept {
    int result = 0;
    if (fd_ != -1) {
      result = ::close(fd_);
    }
    fd_ = new_fd;
    return result;
  }

 private:
  int fd_ = -1;
};

}  // namespace arcserve::net
