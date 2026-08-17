#pragma once

#include <cstdint>
#include <string>

namespace pebbledb::util {

// Fixed-width little-endian integer encode/decode helpers used by every
// PebbleDB on-disk binary format (WAL records today; SSTable blocks and
// the manifest in later phases). Bytes are shifted out/in explicitly
// rather than via a raw memcpy/reinterpret_cast of a native integer, so
// the on-disk layout is identical regardless of host endianness or
// struct-padding rules — both of those would otherwise silently make the
// format non-portable and depend on undefined behavior (strict aliasing).
// Little-endian was picked arbitrarily (spec decision D10-equivalent is
// "exact encoding is a design decision"); see
// docs/decisions/0007-wal-record-binary-format.md.

void PutFixed32(std::string* dst, std::uint32_t value);
void PutFixed64(std::string* dst, std::uint64_t value);

// Decodes a little-endian 32/64-bit value from the 4/8 bytes starting at
// `ptr`. The caller must guarantee at least 4/8 readable bytes at `ptr`;
// these helpers do no bounds checking themselves (that happens one layer
// up, e.g. wal::DecodeHeader checking the buffer length before calling
// these on fixed offsets within it).
std::uint32_t DecodeFixed32(const char* ptr);
std::uint64_t DecodeFixed64(const char* ptr);

}  // namespace pebbledb::util
