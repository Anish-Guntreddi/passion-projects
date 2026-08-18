#include "pebbledb/manifest/format.hpp"

#include "pebbledb/util/coding.hpp"
#include "pebbledb/util/crc32c.hpp"

namespace pebbledb::manifest {

namespace {

// Byte offsets within the fixed-size header, mirroring wal/record.cpp and
// sstable/format.cpp's convention of keeping raw offsets private to this
// translation unit -- see docs/file-format.md for the authoritative
// layout table.
constexpr std::size_t kChecksumOffset = 0;
constexpr std::size_t kMagicOffset = 4;
constexpr std::size_t kFormatVersionOffset = 8;
constexpr std::size_t kNextFileNumberOffset = 9;
constexpr std::size_t kWalFileNumberOffset = 17;
constexpr std::size_t kLastSequenceOffset = 25;
constexpr std::size_t kNumTablesOffset = 33;
static_assert(kNumTablesOffset + 4 == kHeaderSize, "manifest header layout/size mismatch");

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
    case DecodeStatus::kBadTableCount:
      return "BadTableCount";
    case DecodeStatus::kBadLength:
      return "BadLength";
    case DecodeStatus::kChecksumMismatch:
      return "ChecksumMismatch";
  }
  return "Unknown";
}

void EncodeManifest(const ManifestData& data, std::string* dst) {
  std::string body;
  body.reserve(kHeaderSize - 4 + data.tables.size() * kTableEntrySize);
  util::PutFixed32(&body, kMagic);
  body.push_back(static_cast<char>(kFormatVersion));
  util::PutFixed64(&body, data.next_file_number);
  util::PutFixed64(&body, data.wal_file_number);
  util::PutFixed64(&body, data.last_sequence);
  util::PutFixed32(&body, static_cast<std::uint32_t>(data.tables.size()));
  for (const TableFileInfo& table : data.tables) {
    util::PutFixed64(&body, table.file_number);
    util::PutFixed32(&body, table.level);
  }

  const std::uint32_t checksum = util::Crc32c(body);
  dst->reserve(dst->size() + 4 + body.size());
  util::PutFixed32(dst, checksum);
  dst->append(body);
}

DecodeStatus DecodeManifest(std::string_view bytes, ManifestData* out) {
  if (bytes.size() < kHeaderSize) {
    return DecodeStatus::kTruncated;
  }

  const std::uint32_t magic = util::DecodeFixed32(bytes.data() + kMagicOffset);
  if (magic != kMagic) {
    return DecodeStatus::kBadMagic;
  }

  const auto format_version = static_cast<std::uint8_t>(bytes[kFormatVersionOffset]);
  if (format_version == 0 || format_version > kFormatVersion) {
    return DecodeStatus::kUnsupportedVersion;
  }

  // num_tables is not yet checksum-verified at this point -- bound it
  // against kMaxTableCount before using it to compute an expected total
  // size, exactly the same "sanity-bound a length field before trusting
  // it" discipline as wal::kMaxKeyLength/kMaxValueLength (ADR 0006). This
  // also prevents `num_tables * kTableEntrySize` below from being able to
  // overflow std::size_t.
  const std::uint32_t num_tables = util::DecodeFixed32(bytes.data() + kNumTablesOffset);
  if (num_tables > kMaxTableCount) {
    return DecodeStatus::kBadTableCount;
  }

  const std::size_t expected_size = kHeaderSize + static_cast<std::size_t>(num_tables) * kTableEntrySize;
  if (bytes.size() < expected_size) {
    return DecodeStatus::kTruncated;
  }
  if (bytes.size() > expected_size) {
    // The manifest is a whole-file format, not an appendable log (unlike
    // the WAL) -- trailing bytes beyond the last table entry are exactly
    // as invalid as a short file, never "extra data to ignore". See ADR
    // 0012.
    return DecodeStatus::kBadLength;
  }

  // Now that `expected_size` is trusted, the checksum can cover exactly
  // the real content: [magic, end of the last table entry).
  const std::uint32_t stored_checksum = util::DecodeFixed32(bytes.data() + kChecksumOffset);
  const std::uint32_t computed =
      util::Crc32c(bytes.data() + kMagicOffset, expected_size - kMagicOffset);
  if (computed != stored_checksum) {
    return DecodeStatus::kChecksumMismatch;
  }

  out->next_file_number = util::DecodeFixed64(bytes.data() + kNextFileNumberOffset);
  out->wal_file_number = util::DecodeFixed64(bytes.data() + kWalFileNumberOffset);
  out->last_sequence = util::DecodeFixed64(bytes.data() + kLastSequenceOffset);
  out->tables.clear();
  out->tables.reserve(num_tables);
  std::size_t offset = kHeaderSize;
  for (std::uint32_t i = 0; i < num_tables; ++i) {
    TableFileInfo table;
    table.file_number = util::DecodeFixed64(bytes.data() + offset);
    table.level = util::DecodeFixed32(bytes.data() + offset + 8);
    out->tables.push_back(table);
    offset += kTableEntrySize;
  }
  return DecodeStatus::kOk;
}

}  // namespace pebbledb::manifest
