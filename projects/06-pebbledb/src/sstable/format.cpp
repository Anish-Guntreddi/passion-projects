#include "pebbledb/sstable/format.hpp"

#include "pebbledb/util/coding.hpp"
#include "pebbledb/util/crc32c.hpp"

namespace pebbledb::sstable {

namespace {

// Byte offsets within the kFooterSize-byte footer. Kept private to this
// translation unit, mirroring wal/record.cpp's convention -- callers only
// ever interact with Footer/EncodeFooter/DecodeFooter, never raw offsets;
// see docs/file-format.md for the authoritative layout table.
constexpr std::size_t kFooterChecksumOffset = 0;
constexpr std::size_t kFooterMagicOffset = 4;
constexpr std::size_t kFooterFormatVersionOffset = 8;
constexpr std::size_t kFooterIndexOffsetOffset = 9;
constexpr std::size_t kFooterIndexSizeOffset = 17;
constexpr std::size_t kFooterFilterOffsetOffset = 25;
constexpr std::size_t kFooterFilterSizeOffset = 33;
constexpr std::size_t kFooterNumEntriesOffset = 41;
static_assert(kFooterNumEntriesOffset + 8 == kFooterSize, "footer layout/size mismatch");

}  // namespace

const char* DecodeStatusName(DecodeStatus status) {
  switch (status) {
    case DecodeStatus::kOk:
      return "Ok";
    case DecodeStatus::kTruncated:
      return "Truncated";
    case DecodeStatus::kBadMagic:
      return "BadMagic";
    case DecodeStatus::kUnsupportedVersion:
      return "UnsupportedVersion";
    case DecodeStatus::kChecksumMismatch:
      return "ChecksumMismatch";
    case DecodeStatus::kBadEntryType:
      return "BadEntryType";
    case DecodeStatus::kUnsupportedCompression:
      return "UnsupportedCompression";
  }
  return "Unknown";
}

void EncodeEntry(const std::string& key, std::uint64_t sequence, EntryType type,
                  const std::string& value, std::string* dst) {
  const bool is_tombstone = (type == EntryType::kTombstone);
  const std::uint32_t key_length = static_cast<std::uint32_t>(key.size());
  const std::uint32_t value_length = is_tombstone ? 0u : static_cast<std::uint32_t>(value.size());

  dst->push_back(static_cast<char>(type));
  util::PutFixed64(dst, sequence);
  util::PutFixed32(dst, key_length);
  util::PutFixed32(dst, value_length);
  dst->append(key.data(), key.size());
  if (!is_tombstone) {
    dst->append(value.data(), value.size());
  }
}

DecodeStatus DecodeEntry(std::string_view* input, Entry* out) {
  if (input->size() < kEntryHeaderSize) {
    return DecodeStatus::kTruncated;
  }

  const auto type_byte = static_cast<std::uint8_t>((*input)[0]);
  if (type_byte != static_cast<std::uint8_t>(EntryType::kValue) &&
      type_byte != static_cast<std::uint8_t>(EntryType::kTombstone)) {
    return DecodeStatus::kBadEntryType;
  }

  const std::uint64_t sequence = util::DecodeFixed64(input->data() + 1);
  const std::uint32_t key_length = util::DecodeFixed32(input->data() + 9);
  const std::uint32_t value_length = util::DecodeFixed32(input->data() + 13);

  // See DecodeEntry's header comment: no separate sanity-bound check on
  // key_length/value_length is needed -- this comparison against the
  // already-bounded in-memory `input` buffer is the only guard required.
  const std::size_t total =
      kEntryHeaderSize + static_cast<std::size_t>(key_length) + value_length;
  if (input->size() < total) {
    return DecodeStatus::kTruncated;
  }

  out->type = static_cast<EntryType>(type_byte);
  out->sequence = sequence;
  out->key.assign(input->data() + kEntryHeaderSize, key_length);
  out->value.assign(input->data() + kEntryHeaderSize + key_length, value_length);
  input->remove_prefix(total);
  return DecodeStatus::kOk;
}

void EncodeIndexEntry(const IndexEntry& entry, std::string* dst) {
  util::PutFixed32(dst, static_cast<std::uint32_t>(entry.last_key.size()));
  dst->append(entry.last_key);
  util::PutFixed64(dst, entry.handle.offset);
  util::PutFixed64(dst, entry.handle.size);
}

DecodeStatus DecodeIndexEntry(std::string_view* input, IndexEntry* out) {
  constexpr std::size_t kFixedPart = 4 + 8 + 8;  // key_length + offset + size
  if (input->size() < 4) {
    return DecodeStatus::kTruncated;
  }
  const std::uint32_t key_length = util::DecodeFixed32(input->data());
  const std::size_t total = kFixedPart + key_length;
  if (input->size() < total) {
    return DecodeStatus::kTruncated;
  }

  out->last_key.assign(input->data() + 4, key_length);
  out->handle.offset = util::DecodeFixed64(input->data() + 4 + key_length);
  out->handle.size = util::DecodeFixed64(input->data() + 4 + key_length + 8);
  input->remove_prefix(total);
  return DecodeStatus::kOk;
}

std::size_t FinishBlock(const std::string& block_contents, std::string* dst) {
  const std::size_t start = dst->size();
  dst->append(block_contents);
  dst->push_back(static_cast<char>(CompressionType::kNone));

  std::uint32_t crc = util::Crc32c(block_contents);
  crc = util::ExtendCrc32c(crc, dst->data() + dst->size() - 1, 1);  // the compression byte
  util::PutFixed32(dst, crc);

  return dst->size() - start;
}

DecodeStatus VerifyAndStripBlockTrailer(std::string_view raw_block,
                                         std::string_view* entries_view) {
  if (raw_block.size() < kBlockTrailerSize) {
    return DecodeStatus::kTruncated;
  }
  const std::size_t entries_len = raw_block.size() - kBlockTrailerSize;
  const std::string_view entries = raw_block.substr(0, entries_len);

  const auto compression_byte = static_cast<std::uint8_t>(raw_block[entries_len]);
  if (compression_byte != static_cast<std::uint8_t>(CompressionType::kNone)) {
    return DecodeStatus::kUnsupportedCompression;
  }

  const std::uint32_t stored_checksum =
      util::DecodeFixed32(raw_block.data() + entries_len + 1);
  std::uint32_t computed = util::Crc32c(entries.data(), entries.size());
  computed = util::ExtendCrc32c(computed, raw_block.data() + entries_len, 1);
  if (computed != stored_checksum) {
    return DecodeStatus::kChecksumMismatch;
  }

  *entries_view = entries;
  return DecodeStatus::kOk;
}

void EncodeFooter(const Footer& footer, std::string* dst) {
  std::string body;
  body.reserve(kFooterSize - 4);
  util::PutFixed32(&body, kMagic);
  body.push_back(static_cast<char>(footer.format_version));
  util::PutFixed64(&body, footer.index_handle.offset);
  util::PutFixed64(&body, footer.index_handle.size);
  util::PutFixed64(&body, footer.filter_handle.offset);
  util::PutFixed64(&body, footer.filter_handle.size);
  util::PutFixed64(&body, footer.num_entries);

  const std::uint32_t checksum = util::Crc32c(body);
  dst->reserve(dst->size() + 4 + body.size());
  util::PutFixed32(dst, checksum);
  dst->append(body);
}

DecodeStatus DecodeFooter(std::string_view footer_bytes, Footer* out) {
  if (footer_bytes.size() < kFooterSize) {
    return DecodeStatus::kTruncated;
  }

  const std::uint32_t magic = util::DecodeFixed32(footer_bytes.data() + kFooterMagicOffset);
  if (magic != kMagic) {
    return DecodeStatus::kBadMagic;
  }

  const auto format_version = static_cast<std::uint8_t>(footer_bytes[kFooterFormatVersionOffset]);
  if (format_version == 0 || format_version > kFormatVersion) {
    return DecodeStatus::kUnsupportedVersion;
  }

  const std::uint32_t stored_checksum =
      util::DecodeFixed32(footer_bytes.data() + kFooterChecksumOffset);
  const std::uint32_t computed =
      util::Crc32c(footer_bytes.data() + kFooterMagicOffset, kFooterSize - kFooterMagicOffset);
  if (computed != stored_checksum) {
    return DecodeStatus::kChecksumMismatch;
  }

  out->stored_checksum = stored_checksum;
  out->format_version = format_version;
  out->index_handle.offset = util::DecodeFixed64(footer_bytes.data() + kFooterIndexOffsetOffset);
  out->index_handle.size = util::DecodeFixed64(footer_bytes.data() + kFooterIndexSizeOffset);
  out->filter_handle.offset = util::DecodeFixed64(footer_bytes.data() + kFooterFilterOffsetOffset);
  out->filter_handle.size = util::DecodeFixed64(footer_bytes.data() + kFooterFilterSizeOffset);
  out->num_entries = util::DecodeFixed64(footer_bytes.data() + kFooterNumEntriesOffset);
  return DecodeStatus::kOk;
}

}  // namespace pebbledb::sstable
