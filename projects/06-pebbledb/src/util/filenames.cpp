#include "pebbledb/util/filenames.hpp"

#include <cstdio>

namespace pebbledb::util {

namespace {

std::string NumberedFileName(const std::string& dbname, std::uint64_t file_number,
                              const char* extension) {
  char number_buf[32];
  // Zero-padded to 6 digits -- see filenames.hpp. %06llu (not %06lu):
  // std::uint64_t is `unsigned long long` on the LP64 targets this
  // project builds for (Linux/WSL2), and mismatching the printf length
  // modifier is undefined behavior, not merely a style nit.
  std::snprintf(number_buf, sizeof(number_buf), "%06llu",
                static_cast<unsigned long long>(file_number));
  return dbname + "/" + number_buf + extension;
}

}  // namespace

std::string WalFileName(const std::string& dbname, std::uint64_t file_number) {
  return NumberedFileName(dbname, file_number, ".wal");
}

std::string SstableFileName(const std::string& dbname, std::uint64_t file_number) {
  return NumberedFileName(dbname, file_number, ".sst");
}

}  // namespace pebbledb::util
