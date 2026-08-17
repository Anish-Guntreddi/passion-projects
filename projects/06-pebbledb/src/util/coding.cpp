#include "pebbledb/util/coding.hpp"

namespace pebbledb::util {

void PutFixed32(std::string* dst, std::uint32_t value) {
  char buf[4];
  buf[0] = static_cast<char>(value & 0xFFu);
  buf[1] = static_cast<char>((value >> 8) & 0xFFu);
  buf[2] = static_cast<char>((value >> 16) & 0xFFu);
  buf[3] = static_cast<char>((value >> 24) & 0xFFu);
  dst->append(buf, sizeof(buf));
}

void PutFixed64(std::string* dst, std::uint64_t value) {
  char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
  }
  dst->append(buf, sizeof(buf));
}

std::uint32_t DecodeFixed32(const char* ptr) {
  const auto* b = reinterpret_cast<const unsigned char*>(ptr);
  return (static_cast<std::uint32_t>(b[0])) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t DecodeFixed64(const char* ptr) {
  const auto* b = reinterpret_cast<const unsigned char*>(ptr);
  std::uint64_t result = 0;
  for (int i = 7; i >= 0; --i) {
    result = (result << 8) | static_cast<std::uint64_t>(b[i]);
  }
  return result;
}

}  // namespace pebbledb::util
