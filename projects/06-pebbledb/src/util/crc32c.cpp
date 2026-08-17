#include "pebbledb/util/crc32c.hpp"

#include <array>

namespace pebbledb::util {

namespace {

// Reversed-bit CRC-32C polynomial (Castagnoli, 0x1EDC6F41 in normal form).
constexpr std::uint32_t kPoly = 0x82F63B78u;

constexpr std::array<std::uint32_t, 256> BuildTable() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t crc = i;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1u) ? (kPoly ^ (crc >> 1)) : (crc >> 1);
    }
    table[i] = crc;
  }
  return table;
}

constexpr std::array<std::uint32_t, 256> kTable = BuildTable();

}  // namespace

std::uint32_t ExtendCrc32c(std::uint32_t partial_crc, const void* data, std::size_t length) {
  std::uint32_t crc = partial_crc ^ 0xFFFFFFFFu;
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < length; ++i) {
    crc = kTable[(crc ^ bytes[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

std::uint32_t Crc32c(const void* data, std::size_t length) { return ExtendCrc32c(0, data, length); }

std::uint32_t Crc32c(std::string_view data) { return Crc32c(data.data(), data.size()); }

}  // namespace pebbledb::util
