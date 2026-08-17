#include "pebbledb/wal/reader.hpp"

namespace pebbledb::wal {

const char* ReadResultName(ReadResult result) {
  switch (result) {
    case ReadResult::kOk:
      return "Ok";
    case ReadResult::kEndOfLog:
      return "EndOfLog";
    case ReadResult::kTruncated:
      return "Truncated";
    case ReadResult::kCorruption:
      return "Corruption";
    case ReadResult::kUnsupportedVersion:
      return "UnsupportedVersion";
  }
  return "Unknown";
}

Status Reader::Open(const std::string& path, std::unique_ptr<Reader>* out) {
  std::unique_ptr<util::PosixFile> file;
  Status s = util::PosixFile::OpenRead(path, &file);
  if (!s.ok()) {
    return s;
  }
  out->reset(new Reader(std::move(file)));
  return Status::OK();
}

ReadResult Reader::ReadNext(Record* record) {
  if (exhausted_) {
    return terminal_result_;
  }

  std::string header_buf;
  bool header_short = false;
  Status s = file_->Read(kHeaderSize, &header_buf, &header_short);
  if (!s.ok()) {
    exhausted_ = true;
    terminal_result_ = ReadResult::kCorruption;
    return terminal_result_;
  }
  if (header_buf.empty()) {
    // Clean EOF exactly at a record boundary: nothing was even started.
    exhausted_ = true;
    terminal_result_ = ReadResult::kEndOfLog;
    return terminal_result_;
  }
  if (header_short) {
    // Some bytes were present but not a full header: a torn write.
    exhausted_ = true;
    terminal_result_ = ReadResult::kTruncated;
    return terminal_result_;
  }

  DecodedHeader decoded;
  DecodeStatus header_status = DecodeHeader(header_buf, &decoded);
  if (header_status != DecodeStatus::kOk) {
    exhausted_ = true;
    terminal_result_ = (header_status == DecodeStatus::kUnsupportedVersion)
                            ? ReadResult::kUnsupportedVersion
                            : ReadResult::kCorruption;
    return terminal_result_;
  }

  const std::size_t payload_len =
      static_cast<std::size_t>(decoded.key_length) + decoded.value_length;
  std::string payload_buf;
  bool payload_short = false;
  if (payload_len > 0) {
    s = file_->Read(payload_len, &payload_buf, &payload_short);
    if (!s.ok()) {
      exhausted_ = true;
      terminal_result_ = ReadResult::kCorruption;
      return terminal_result_;
    }
  }
  if (payload_short) {
    exhausted_ = true;
    terminal_result_ = ReadResult::kTruncated;
    return terminal_result_;
  }

  DecodeStatus body_status = DecodeBody(decoded, header_buf, payload_buf, record);
  if (body_status != DecodeStatus::kOk) {
    exhausted_ = true;
    terminal_result_ = ReadResult::kCorruption;
    return terminal_result_;
  }

  return ReadResult::kOk;
}

Reader::ReplayResult Reader::Replay(const std::function<void(const Record&)>& on_record) {
  ReplayResult result;
  Record record;
  while (true) {
    ReadResult r = ReadNext(&record);
    if (r != ReadResult::kOk) {
      result.terminal = r;
      break;
    }
    if (on_record) {
      on_record(record);
    }
    ++result.records_replayed;
  }
  return result;
}

}  // namespace pebbledb::wal
