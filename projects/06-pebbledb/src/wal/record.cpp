#include "pebbledb/wal/record.hpp"

#include "pebbledb/util/coding.hpp"
#include "pebbledb/util/crc32c.hpp"

namespace pebbledb::wal {

namespace {

// Byte offsets within the kHeaderSize-byte fixed header. Kept private to
// this translation unit: callers only ever interact with DecodedHeader /
// EncodeRecord, never raw offsets — see docs/file-format.md for the
// authoritative layout table.
constexpr std::size_t kChecksumOffset = 0;
constexpr std::size_t kMagicOffset = 4;
constexpr std::size_t kFormatVersionOffset = 8;
constexpr std::size_t kOpOffset = 9;
constexpr std::size_t kSequenceOffset = 10;
constexpr std::size_t kKeyLengthOffset = 18;
constexpr std::size_t kValueLengthOffset = 22;
static_assert(kValueLengthOffset + 4 == kHeaderSize, "header layout/size mismatch");

}  // namespace

const char* DecodeStatusName(DecodeStatus status) {
  switch (status) {
    case DecodeStatus::kOk:
      return "Ok";
    case DecodeStatus::kTruncatedHeader:
      return "TruncatedHeader";
    case DecodeStatus::kBadMagic:
      return "BadMagic";
    case DecodeStatus::kUnsupportedVersion:
      return "UnsupportedVersion";
    case DecodeStatus::kBadLength:
      return "BadLength";
    case DecodeStatus::kTruncatedPayload:
      return "TruncatedPayload";
    case DecodeStatus::kBadOperation:
      return "BadOperation";
    case DecodeStatus::kChecksumMismatch:
      return "ChecksumMismatch";
  }
  return "Unknown";
}

void EncodeRecord(const Record& record, std::string* dst) {
  const std::uint32_t key_length = static_cast<std::uint32_t>(record.key.size());
  // A Delete record's value is never persisted, regardless of what the
  // caller put in record.value — see record.hpp / docs/file-format.md.
  const bool is_delete = (record.type == RecordType::kDelete);
  const std::uint32_t value_length =
      is_delete ? 0u : static_cast<std::uint32_t>(record.value.size());

  // Build the checksummed region (everything after the checksum field)
  // first, so the checksum can be computed over it in one pass.
  std::string body;
  body.reserve(kHeaderSize - 4 + key_length + value_length);
  util::PutFixed32(&body, kMagic);
  body.push_back(static_cast<char>(kFormatVersion));
  body.push_back(static_cast<char>(record.type));
  util::PutFixed64(&body, record.sequence);
  util::PutFixed32(&body, key_length);
  util::PutFixed32(&body, value_length);
  body.append(record.key.data(), record.key.size());
  if (!is_delete) {
    body.append(record.value.data(), record.value.size());
  }

  const std::uint32_t checksum = util::Crc32c(body);

  dst->reserve(dst->size() + 4 + body.size());
  util::PutFixed32(dst, checksum);
  dst->append(body);
}

DecodeStatus DecodeHeader(std::string_view header, DecodedHeader* out) {
  if (header.size() < kHeaderSize) {
    return DecodeStatus::kTruncatedHeader;
  }

  const std::uint32_t magic = util::DecodeFixed32(header.data() + kMagicOffset);
  if (magic != kMagic) {
    return DecodeStatus::kBadMagic;
  }

  const auto format_version = static_cast<std::uint8_t>(header[kFormatVersionOffset]);
  if (format_version == 0 || format_version > kFormatVersion) {
    return DecodeStatus::kUnsupportedVersion;
  }

  const std::uint32_t key_length = util::DecodeFixed32(header.data() + kKeyLengthOffset);
  const std::uint32_t value_length = util::DecodeFixed32(header.data() + kValueLengthOffset);
  if (key_length > kMaxKeyLength || value_length > kMaxValueLength) {
    return DecodeStatus::kBadLength;
  }

  out->stored_checksum = util::DecodeFixed32(header.data() + kChecksumOffset);
  out->format_version = format_version;
  out->op_byte = static_cast<std::uint8_t>(header[kOpOffset]);
  out->sequence = util::DecodeFixed64(header.data() + kSequenceOffset);
  out->key_length = key_length;
  out->value_length = value_length;
  return DecodeStatus::kOk;
}

DecodeStatus DecodeBody(const DecodedHeader& decoded_header, std::string_view header,
                         std::string_view payload, Record* record) {
  const std::size_t expected_payload =
      static_cast<std::size_t>(decoded_header.key_length) + decoded_header.value_length;
  if (payload.size() < expected_payload) {
    return DecodeStatus::kTruncatedPayload;
  }

  // Checksum covers everything after the checksum field: the rest of the
  // header (magic..value_length) plus the payload.
  std::uint32_t computed = util::Crc32c(header.data() + kMagicOffset, kHeaderSize - kMagicOffset);
  computed = util::ExtendCrc32c(computed, payload.data(), payload.size());
  if (computed != decoded_header.stored_checksum) {
    return DecodeStatus::kChecksumMismatch;
  }

  if (decoded_header.op_byte != static_cast<std::uint8_t>(RecordType::kPut) &&
      decoded_header.op_byte != static_cast<std::uint8_t>(RecordType::kDelete)) {
    return DecodeStatus::kBadOperation;
  }

  record->sequence = decoded_header.sequence;
  record->type = static_cast<RecordType>(decoded_header.op_byte);
  record->key.assign(payload.data(), decoded_header.key_length);
  record->value.assign(payload.data() + decoded_header.key_length, decoded_header.value_length);
  return DecodeStatus::kOk;
}

}  // namespace pebbledb::wal
