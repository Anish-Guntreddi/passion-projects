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

}  // namespace pebbledb::util
