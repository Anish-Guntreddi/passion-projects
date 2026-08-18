#include "pebbledb/sstable/writer.hpp"

#include "pebbledb/bloom/bloom_filter.hpp"

namespace pebbledb::sstable {

Status Writer::Open(const std::string& path, const Options& options,
                     std::unique_ptr<Writer>* out) {
  std::unique_ptr<util::PosixFile> file;
  Status s = util::PosixFile::OpenNew(path, &file);
  if (!s.ok()) {
    return s;
  }
  out->reset(new Writer(std::move(file), options));
  return Status::OK();
}

Status Writer::Add(Slice key, Slice value, std::uint64_t sequence, EntryType type) {
  if (finished_) {
    return Status::InvalidArgument("Writer::Add() called after Finish()");
  }
  if (has_added_any_ && key <= Slice(last_key_added_)) {
    return Status::InvalidArgument(
        "sstable::Writer requires keys to be added in strictly increasing order");
  }

  const bool is_tombstone = (type == EntryType::kTombstone);
  EncodeEntry(std::string(key), sequence, type, is_tombstone ? std::string() : std::string(value),
              &current_block_entries_);
  last_key_in_current_block_.assign(key.data(), key.size());
  current_block_has_entries_ = true;

  if (options_.filter_bits_per_key > 0) {
    keys_for_filter_.emplace_back(key);
  }

  last_key_added_.assign(key.data(), key.size());
  has_added_any_ = true;
  ++entries_written_;

  if (current_block_entries_.size() >= options_.target_block_size_bytes) {
    Status s = FlushCurrentBlock();
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status Writer::FlushCurrentBlock() {
  if (!current_block_has_entries_) {
    return Status::OK();
  }

  std::string block_bytes;
  const std::size_t written = FinishBlock(current_block_entries_, &block_bytes);
  Status s = file_->Append(block_bytes);
  if (!s.ok()) {
    return s;
  }

  IndexEntry index_entry;
  index_entry.last_key = last_key_in_current_block_;
  index_entry.handle.offset = file_offset_;
  index_entry.handle.size = written;
  EncodeIndexEntry(index_entry, &index_block_);

  file_offset_ += written;
  current_block_entries_.clear();
  current_block_has_entries_ = false;
  return Status::OK();
}

Status Writer::Finish() {
  if (finished_) {
    return Status::InvalidArgument("Writer::Finish() called more than once");
  }

  Status s = FlushCurrentBlock();
  if (!s.ok()) {
    finished_ = true;  // do not allow further Add()/Finish() after a failed Finish()
    return s;
  }

  // Filter block (spec FR3, roadmap Phase 5; ADR 0009/ADR 0014). Its
  // offset is exactly where the block starts, right after the last data
  // block; a real, non-empty filter is built and written here when
  // filtering is enabled and at least one key was added, otherwise this
  // stays the same zero-length "nothing here" region every pre-Phase-5
  // table already had -- a reader must treat size == 0 as "absent," not
  // as an error, either way.
  BlockHandle filter_handle{file_offset_, 0};
  if (!keys_for_filter_.empty()) {
    const bloom::FilterPolicy policy{options_.filter_bits_per_key};
    const std::string filter_content = bloom::BuildFilter(keys_for_filter_, policy);
    if (!filter_content.empty()) {
      std::string filter_block_bytes;
      const std::size_t filter_written = FinishBlock(filter_content, &filter_block_bytes);
      s = file_->Append(filter_block_bytes);
      if (!s.ok()) {
        finished_ = true;
        return s;
      }
      filter_handle.size = filter_written;
      file_offset_ += filter_written;
    }
  }

  std::string index_block_bytes;
  const std::size_t index_written = FinishBlock(index_block_, &index_block_bytes);
  const BlockHandle index_handle{file_offset_, index_written};
  s = file_->Append(index_block_bytes);
  if (!s.ok()) {
    finished_ = true;
    return s;
  }
  file_offset_ += index_written;

  Footer footer;
  footer.format_version = kFormatVersion;
  footer.index_handle = index_handle;
  footer.filter_handle = filter_handle;
  footer.num_entries = entries_written_;

  std::string footer_bytes;
  EncodeFooter(footer, &footer_bytes);
  s = file_->Append(footer_bytes);
  if (!s.ok()) {
    finished_ = true;
    return s;
  }
  file_offset_ += footer_bytes.size();

  s = file_->Sync();
  finished_ = true;
  return s;
}

}  // namespace pebbledb::sstable
