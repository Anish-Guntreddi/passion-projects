#include "temp_file.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace pebbledb::testutil {

TempFile::TempFile() {
  std::string tmpl = (std::filesystem::temp_directory_path() / "pebbledb_test_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');

  int fd = ::mkstemp(buf.data());
  if (fd < 0) {
    throw std::runtime_error("mkstemp failed while reserving a temp path");
  }
  // mkstemp both creates and opens the file; close it immediately. The
  // (empty) file stays on disk so the name is reserved race-free until
  // the test opens it through the API under test.
  ::close(fd);

  path_ = buf.data();
}

TempFile::~TempFile() {
  std::error_code ec;
  std::filesystem::remove(path_, ec);  // best-effort; ignore failures
}

std::string UniqueNonExistentPath() {
  std::string tmpl =
      (std::filesystem::temp_directory_path() / "pebbledb_test_new_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');

  int fd = ::mkstemp(buf.data());
  if (fd < 0) {
    throw std::runtime_error("mkstemp failed while reserving a unique path");
  }
  ::close(fd);

  std::string path = buf.data();
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (ec) {
    throw std::runtime_error("failed to free reserved path '" + path + "': " + ec.message());
  }
  return path;
}

std::int64_t FileSize(const std::string& path) {
  std::error_code ec;
  auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    return -1;
  }
  return static_cast<std::int64_t>(size);
}

void TruncateFileTo(const std::string& path, std::int64_t length) {
  std::error_code ec;
  std::filesystem::resize_file(path, static_cast<std::uintmax_t>(length), ec);
  if (ec) {
    throw std::runtime_error("resize_file('" + path + "', " + std::to_string(length) +
                              ") failed: " + ec.message());
  }
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open for reading: " + path);
  }
  // Seek-to-end-then-read, rather than the istreambuf_iterator-range
  // constructor: the latter trips GCC's -Wnull-dereference under -O2
  // (Release builds) on the standard library's own gptr()/sbumpc()
  // internals -- a known false positive, not a real bug in this
  // function, but PEBBLEDB_WARNINGS_AS_ERRORS (ADR 0001) means it must
  // still be avoided rather than suppressed.
  f.seekg(0, std::ios::end);
  const auto size = f.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to determine size of: " + path);
  }
  std::string contents(static_cast<std::size_t>(size), '\0');
  f.seekg(0, std::ios::beg);
  if (!contents.empty()) {
    f.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!f) {
      throw std::runtime_error("failed to read: " + path);
    }
  }
  return contents;
}

void WriteWholeFile(const std::string& path, const std::string& contents) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    throw std::runtime_error("failed to open for writing: " + path);
  }
  f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!f) {
    throw std::runtime_error("failed to write: " + path);
  }
}

void CorruptByteAt(const std::string& path, std::int64_t byte_offset, unsigned char mask) {
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open for corruption: " + path);
  }
  f.seekg(byte_offset);
  char c = 0;
  f.read(&c, 1);
  if (!f) {
    throw std::runtime_error("seek/read past end while corrupting byte " +
                              std::to_string(byte_offset) + " of " + path);
  }
  c = static_cast<char>(static_cast<unsigned char>(c) ^ mask);
  f.seekp(byte_offset);
  f.write(&c, 1);
  if (!f) {
    throw std::runtime_error("write failed while corrupting byte " +
                              std::to_string(byte_offset) + " of " + path);
  }
}

TempDir::TempDir() {
  std::string tmpl = (std::filesystem::temp_directory_path() / "pebbledb_test_dir_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');

  if (::mkdtemp(buf.data()) == nullptr) {
    throw std::runtime_error("mkdtemp failed while reserving a temp directory");
  }
  path_ = buf.data();
}

TempDir::~TempDir() {
  std::error_code ec;
  std::filesystem::remove_all(path_, ec);  // best-effort; ignore failures
}

}  // namespace pebbledb::testutil
