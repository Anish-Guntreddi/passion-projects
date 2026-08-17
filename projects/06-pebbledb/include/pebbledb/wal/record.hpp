#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace pebbledb::wal {

// WAL record v1 binary format (spec FR10; full field-by-field
// documentation lives in docs/file-format.md — this comment is a
// pointer, not a duplicate source of truth). Every field is fixed-width
// and little-endian (see util/coding.hpp).
//
//   checksum        uint32   CRC-32C over every byte below (magic
//                            through value), NOT including this field
//                            itself.
//   magic            uint32   constant kMagic; identifies this as a
//                            PebbleDB WAL v1 record.
//   format_version   uint8    constant kFormatVersion (currently 1).
//   op               uint8    kPut (1) or kDelete (2).
//   sequence         uint64   monotonically increasing sequence number
//                            (ADR 0005 / spec decision D2).
//   key_length       uint32   length of `key` in bytes.
//   value_length     uint32   length of `value` in bytes (always 0 for
//                            kDelete; the encoder ignores any value
//                            given for a Delete record).
//   key              bytes    key_length bytes, binary-safe.
//   value            bytes    value_length bytes, binary-safe.
//
// Total fixed header size is kHeaderSize (26) bytes, followed by
// key_length + value_length variable bytes.

enum class RecordType : std::uint8_t {
  kPut = 1,
  kDelete = 2,
};

struct Record {
  std::uint64_t sequence = 0;
  RecordType type = RecordType::kPut;
  std::string key;
  std::string value;  // unused/empty for kDelete
};

inline constexpr std::uint32_t kMagic = 0x50424B31u;  // see docs/file-format.md
inline constexpr std::uint8_t kFormatVersion = 1;

// checksum(4) + magic(4) + format_version(1) + op(1) + sequence(8) +
// key_length(4) + value_length(4)
inline constexpr std::size_t kHeaderSize = 26;

// Sanity bounds on declared lengths, checked before ever attempting to
// read that many payload bytes off disk. These exist so a corrupted
// length field (e.g. a flipped high bit turning a small length into
// ~4 billion) fails fast as a decode error instead of attempting a
// multi-gigabyte read. They are generous relative to any workload this
// project's benchmarks (spec §1.9) exercise, not a claim about a
// production maximum key/value size.
inline constexpr std::uint32_t kMaxKeyLength = 1u << 20;    // 1 MiB
inline constexpr std::uint32_t kMaxValueLength = 1u << 26;  // 64 MiB

// Serializes `record` in the v1 on-disk format described above,
// appending the bytes to `dst` (which is not cleared first — callers
// that want a fresh buffer should clear it themselves).
void EncodeRecord(const Record& record, std::string* dst);

enum class DecodeStatus {
  kOk,
  kTruncatedHeader,     // fewer than kHeaderSize bytes were supplied
  kBadMagic,             // header present but magic != kMagic
  kUnsupportedVersion,  // header present, magic OK, but format_version is
                        // 0 or newer than this reader understands
  kBadLength,           // key_length/value_length exceeds the sanity bound
  kTruncatedPayload,    // payload shorter than key_length + value_length
  kBadOperation,        // op byte is neither kPut nor kDelete
  kChecksumMismatch,    // full record present; checksum did not verify
};

const char* DecodeStatusName(DecodeStatus status);

// Result of parsing just the fixed kHeaderSize-byte header, before the
// variable-length key/value payload has necessarily been read yet (the
// caller needs key_length/value_length to know how many more bytes to
// pull off disk).
struct DecodedHeader {
  std::uint32_t stored_checksum = 0;
  std::uint8_t format_version = 0;
  std::uint8_t op_byte = 0;
  std::uint64_t sequence = 0;
  std::uint32_t key_length = 0;
  std::uint32_t value_length = 0;
};

// Parses `header`, which must be exactly kHeaderSize bytes (fewer yields
// kTruncatedHeader). Validates magic, format_version, and the declared
// length bounds; does NOT validate the checksum (that requires the
// payload too — see DecodeBody) or the op byte (deferred to DecodeBody,
// after checksum verification, so a corrupted op byte is reported
// consistently as a checksum failure rather than racing with it — see
// docs/file-format.md).
DecodeStatus DecodeHeader(std::string_view header, DecodedHeader* out);

// Verifies the checksum over `header` (the same kHeaderSize bytes passed
// to DecodeHeader) and `payload` (which must be exactly
// decoded_header.key_length + decoded_header.value_length bytes; a
// shorter payload yields kTruncatedPayload without attempting checksum
// verification). On a checksum match, validates the op byte and, if
// valid, populates `record` and returns kOk.
DecodeStatus DecodeBody(const DecodedHeader& decoded_header, std::string_view header,
                         std::string_view payload, Record* record);

}  // namespace pebbledb::wal
