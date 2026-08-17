#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace pebbledb::util {

// CRC-32C (Castagnoli polynomial 0x1EDC6F41, reversed-bit form
// 0x82F63B78) — the checksum used by every PebbleDB on-disk record (spec
// FR10; docs/file-format.md). Picked over classic CRC-32 (the zlib/gzip
// polynomial, 0xEDB88320) because it has strictly better error-detection
// properties for short records (better Hamming distance at the record
// sizes a WAL/SSTable block actually uses) and is the same polynomial
// widely used for exactly this purpose elsewhere (iSCSI/SCTP, ext4
// metadata, and the SSE4.2 CRC32 instruction all use CRC-32C).
//
// Table-based ("slice-by-1") implementation: a 256-entry lookup table,
// built once at compile time via constexpr evaluation, processes one
// input byte per table lookup. This is the standard, well-documented
// software CRC construction (not vendored from any external library) —
// see crc32c.cpp for the table-generation code and
// docs/decisions/0007-wal-record-binary-format.md for why a checksum is
// used at all and why this specific polynomial.

// Computes the CRC-32C of `length` bytes at `data`, starting from the
// standard initial state (equivalent to ExtendCrc32c(0, data, length)).
std::uint32_t Crc32c(const void* data, std::size_t length);
std::uint32_t Crc32c(std::string_view data);

// Extends a previously computed CRC-32C value with `length` more bytes at
// `data`. This lets a record's checksum be computed across
// non-contiguous buffers (a fixed header, then a key, then a value)
// without concatenating them into one buffer first:
//
//   std::uint32_t crc = Crc32c(header);
//   crc = ExtendCrc32c(crc, key.data(), key.size());
//   crc = ExtendCrc32c(crc, value.data(), value.size());
//
// is exactly equal to Crc32c(header + key + value) computed on the
// concatenation. Passing 0 as `partial_crc` is equivalent to starting a
// fresh checksum (Crc32c(...) is defined in terms of this property).
std::uint32_t ExtendCrc32c(std::uint32_t partial_crc, const void* data, std::size_t length);

}  // namespace pebbledb::util
