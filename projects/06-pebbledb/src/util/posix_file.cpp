#include "pebbledb/util/posix_file.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>

namespace pebbledb::util {

namespace {
std::string ErrnoMessage(const std::string& what) { return what + ": " + std::strerror(errno); }

// Fsyncs the directory containing `path`. fsync() on a file descriptor
// only guarantees that *file's* contents are durable; on POSIX
// filesystems it does not by itself guarantee that the directory entry
// pointing to a brand-new file survives a crash -- that requires a
// separate fsync() of the containing directory. See docs/durability.md,
// "File creation durability: directory entries", and ADR 0006.
Status SyncContainingDirectory(const std::string& path) {
  std::filesystem::path parent = std::filesystem::path(path).parent_path();
  const std::string dir = parent.empty() ? "." : parent.string();

  int dir_fd = ::open(dir.c_str(), O_RDONLY);
  if (dir_fd < 0) {
    return Status::IOError(ErrnoMessage("open (directory) failed for '" + dir + "'"));
  }
  int rc;
  do {
    rc = ::fsync(dir_fd);
  } while (rc != 0 && errno == EINTR);
  const int saved_errno = errno;
  ::close(dir_fd);
  if (rc != 0) {
    errno = saved_errno;
    return Status::IOError(ErrnoMessage("fsync (directory) failed for '" + dir + "'"));
  }
  return Status::OK();
}
}  // namespace

Status PosixFile::OpenAppend(const std::string& path, std::unique_ptr<PosixFile>* out) {
  // Stat before O_CREAT so we know whether this call is the one creating
  // a brand-new directory entry (which needs a directory fsync, below) or
  // just reopening an already-existing WAL file for continued appending
  // (which does not -- no new directory entry was created).
  struct stat existing_stat {};
  const bool existed_before = (::stat(path.c_str(), &existing_stat) == 0);

  int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
  if (fd < 0) {
    return Status::IOError(ErrnoMessage("open (append) failed for '" + path + "'"));
  }

  if (!existed_before) {
    // A new directory entry was just created for `path`. Fsync the
    // directory *before* reporting Open() successful, so that by the
    // time a caller's first AddRecord()/Sync() acknowledges a write
    // (making the file's own bytes durable), the directory entry needed
    // to reach that file after a crash is already durable too -- see
    // docs/durability.md.
    Status dir_status = SyncContainingDirectory(path);
    if (!dir_status.ok()) {
      ::close(fd);
      return dir_status;
    }
  }

  out->reset(new PosixFile(fd));
  return Status::OK();
}

Status PosixFile::OpenNew(const std::string& path, std::unique_ptr<PosixFile>* out) {
  // A path that already has real (non-zero-byte) content is refused
  // outright: a writer built on OpenNew() assumes it is starting from
  // file offset 0, so it must never be pointed at a file that already
  // has bytes in it (see OpenNew()'s header comment and ADR 0011). A
  // zero-byte file, though, is not "existing content" in any sense that
  // matters here -- it is indistinguishable from "does not exist yet"
  // for every purpose this call cares about, so it is accepted exactly
  // like a genuinely-absent path. (This is also what lets callers use a
  // race-free-but-empty pre-reserved path -- e.g. tests/support's
  // TempFile -- with an OpenNew()-based writer with no special-casing.)
  struct stat existing_stat {};
  const bool existed_before = (::stat(path.c_str(), &existing_stat) == 0);
  if (existed_before && existing_stat.st_size != 0) {
    return Status::IOError("OpenNew: refusing to open '" + path + "': path already has " +
                            std::to_string(existing_stat.st_size) +
                            " byte(s) of content (expected an absent or empty path)");
  }

  // O_TRUNC is defense-in-depth, not the primary guarantee -- the check
  // above is what actually rejects real pre-existing content. This just
  // guarantees a genuinely zero-byte starting point even across the
  // stat()-then-open() window above, which (like the same window in
  // OpenAppend(), above) is not a concern this single-writer-per-file
  // project defends against concurrently (ADR 0003). No O_APPEND: this
  // fd is for one sequential writer tracking its own offset from 0, not
  // the "safe under concurrent appenders" scenario O_APPEND exists for
  // on the WAL path.
  int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    return Status::IOError(ErrnoMessage("open (new) failed for '" + path + "'"));
  }

  if (!existed_before) {
    // A new directory entry was just created for `path` -- fsync the
    // containing directory before reporting success, same as
    // OpenAppend()'s "did this call create a new entry" case.
    Status dir_status = SyncContainingDirectory(path);
    if (!dir_status.ok()) {
      ::close(fd);
      return dir_status;
    }
  }

  out->reset(new PosixFile(fd));
  return Status::OK();
}

Status PosixFile::OpenRead(const std::string& path, std::unique_ptr<PosixFile>* out) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return Status::IOError(ErrnoMessage("open (read) failed for '" + path + "'"));
  }
  out->reset(new PosixFile(fd));
  return Status::OK();
}

PosixFile::~PosixFile() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

Status PosixFile::Append(std::string_view data) {
  std::size_t total = 0;
  while (total < data.size()) {
    ssize_t written = ::write(fd_, data.data() + total, data.size() - total);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return Status::IOError(ErrnoMessage("write failed"));
    }
    total += static_cast<std::size_t>(written);
  }
  return Status::OK();
}

Status PosixFile::Sync() {
  int rc;
  do {
    rc = ::fsync(fd_);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) {
    return Status::IOError(ErrnoMessage("fsync failed"));
  }
  return Status::OK();
}

Status PosixFile::Read(std::size_t n, std::string* out, bool* short_read) {
  out->assign(n, '\0');
  std::size_t total = 0;
  while (total < n) {
    ssize_t got = ::read(fd_, out->data() + total, n - total);
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      out->clear();
      return Status::IOError(ErrnoMessage("read failed"));
    }
    if (got == 0) {
      break;  // EOF
    }
    total += static_cast<std::size_t>(got);
  }
  out->resize(total);
  if (short_read != nullptr) {
    *short_read = (total < n);
  }
  return Status::OK();
}

Status PosixFile::ReadAt(std::uint64_t offset, std::size_t n, std::string* out, bool* short_read) {
  out->assign(n, '\0');
  std::size_t total = 0;
  while (total < n) {
    ssize_t got = ::pread(fd_, out->data() + total, n - total,
                          static_cast<off_t>(offset + static_cast<std::uint64_t>(total)));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      out->clear();
      return Status::IOError(ErrnoMessage("pread failed"));
    }
    if (got == 0) {
      break;  // EOF
    }
    total += static_cast<std::size_t>(got);
  }
  out->resize(total);
  if (short_read != nullptr) {
    *short_read = (total < n);
  }
  return Status::OK();
}

Status PosixFile::Size(std::uint64_t* out) {
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    return Status::IOError(ErrnoMessage("fstat failed"));
  }
  *out = static_cast<std::uint64_t>(st.st_size);
  return Status::OK();
}

}  // namespace pebbledb::util
