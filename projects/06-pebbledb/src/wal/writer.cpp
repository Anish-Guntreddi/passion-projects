#include "pebbledb/wal/writer.hpp"

namespace pebbledb::wal {

Status Writer::Open(const std::string& path, SyncPolicy policy, std::size_t batch_size,
                     std::unique_ptr<Writer>* out) {
  if (policy == SyncPolicy::kBatched && batch_size == 0) {
    return Status::InvalidArgument("batch_size must be >= 1 for SyncPolicy::kBatched");
  }

  std::unique_ptr<util::PosixFile> file;
  Status s = util::PosixFile::OpenAppend(path, &file);
  if (!s.ok()) {
    return s;
  }

  out->reset(new Writer(std::move(file), policy, batch_size));
  return Status::OK();
}

Status Writer::AddRecord(const Record& record) {
  // Reject an oversized key/value *before* encoding or writing anything,
  // rather than letting EncodeRecord() silently truncate it into a
  // uint32_t header field. Without this check, such a record could be
  // written, fsynced, and acknowledged (AddRecord returning OK) here, yet
  // wal::DecodeHeader() enforces these exact bounds on replay and reports
  // any excess as kBadLength -- which Reader maps to kCorruption, halting
  // recovery at that record. An acknowledged write must never become
  // unrecoverable that way -- see docs/durability.md and ADR 0006.
  if (record.key.size() > kMaxKeyLength) {
    return Status::InvalidArgument("WAL record key exceeds kMaxKeyLength");
  }
  const bool is_delete = (record.type == RecordType::kDelete);
  if (!is_delete && record.value.size() > kMaxValueLength) {
    return Status::InvalidArgument("WAL record value exceeds kMaxValueLength");
  }

  std::string buf;
  EncodeRecord(record, &buf);

  Status s = file_->Append(buf);
  if (!s.ok()) {
    return s;
  }
  ++records_written_;
  ++records_since_sync_;

  if (policy_ == SyncPolicy::kEveryWrite) {
    return Sync();
  }
  if (policy_ == SyncPolicy::kBatched && records_since_sync_ >= batch_size_) {
    return Sync();
  }
  return Status::OK();
}

Status Writer::Sync() {
  Status s = file_->Sync();
  if (!s.ok()) {
    return s;
  }
  ++sync_count_;
  records_since_sync_ = 0;
  return Status::OK();
}

}  // namespace pebbledb::wal
