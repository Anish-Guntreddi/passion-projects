#include "pebbledb/sstable/reader.hpp"

#include <algorithm>
#include <utility>

namespace pebbledb::sstable {

namespace {

// Maps a format-level DecodeStatus to a Status. kUnsupportedVersion/
// kUnsupportedCompression are reported as Status::NotSupported (a
// structurally-recognized-but-not-handled value, not evidence of bit
// rot -- mirrors wal::ReadResult::kUnsupportedVersion's rationale);
// every other non-kOk DecodeStatus is genuine corruption.
Status ToStatus(DecodeStatus status, const char* what) {
  if (status == DecodeStatus::kUnsupportedVersion || status == DecodeStatus::kUnsupportedCompression) {
    return Status::NotSupported(std::string(what) + ": " + DecodeStatusName(status));
  }
  return Status::Corruption(std::string(what) + ": " + DecodeStatusName(status));
}

// Rejects a BlockHandle whose [offset, offset + size) range extends
// beyond the file's actual size -- checked *before* the handle is ever
// used to drive a pread()/allocation (util::PosixFile::ReadAt allocates
// `size` bytes up front). Without this, a hand-crafted or bit-flipped
// footer/index entry whose checksum still happens to verify (or, for the
// footer, an offset/size pair the checksum covers but that was corrupted
// together with a matching checksum) could declare an arbitrarily large
// size and crash the process with an uncaught std::bad_alloc instead of
// returning Status::Corruption -- see ADR 0011.
//
// Written as `offset > file_size - size` (after checking `size <=
// file_size`) rather than `offset + size > file_size` so this can never
// itself overflow: `offset` and `size` are attacker/corruption-
// controlled std::uint64_t values, and `offset + size` could wrap around
// to a small number and pass a naive check.
Status ValidateHandleInBounds(const BlockHandle& handle, std::uint64_t file_size, const char* what) {
  if (handle.size > file_size || handle.offset > file_size - handle.size) {
    return Status::Corruption(std::string(what) + ": block handle extends beyond file size");
  }
  return Status::OK();
}

}  // namespace

Status Reader::Open(const std::string& path, std::unique_ptr<Reader>* out) {
  std::unique_ptr<util::PosixFile> file;
  Status s = util::PosixFile::OpenRead(path, &file);
  if (!s.ok()) {
    return s;
  }

  std::uint64_t file_size = 0;
  s = file->Size(&file_size);
  if (!s.ok()) {
    return s;
  }
  if (file_size < kFooterSize) {
    return Status::Corruption("sstable file is smaller than a footer");
  }

  std::string footer_buf;
  bool short_read = false;
  s = file->ReadAt(file_size - kFooterSize, kFooterSize, &footer_buf, &short_read);
  if (!s.ok()) {
    return s;
  }
  if (short_read) {
    return Status::Corruption("short read while reading sstable footer");
  }

  Footer footer;
  DecodeStatus footer_status = DecodeFooter(footer_buf, &footer);
  if (footer_status != DecodeStatus::kOk) {
    return ToStatus(footer_status, "sstable footer");
  }

  std::unique_ptr<Reader> reader(new Reader(std::move(file)));
  reader->footer_ = footer;
  reader->file_size_ = file_size;

  // Validate both footer-declared handles against the file's real size
  // now, at Open() time, rather than letting a corrupted-but-checksum-
  // valid one reach ReadBlock()/ReadAt() later with an out-of-range
  // size. filter_handle is never actually read in this version (always
  // zero-length -- ADR 0009), but a corrupted footer could still declare
  // a bogus non-zero one, so it is validated here for the same reason
  // index_handle is, even though nothing downstream currently reads it.
  s = ValidateHandleInBounds(footer.index_handle, file_size, "sstable index handle");
  if (!s.ok()) {
    return s;
  }
  s = ValidateHandleInBounds(footer.filter_handle, file_size, "sstable filter handle");
  if (!s.ok()) {
    return s;
  }

  if (footer.index_handle.size > 0) {
    std::string raw_index;
    std::string_view index_entries_view;
    s = reader->ReadBlock(footer.index_handle, &raw_index, &index_entries_view);
    if (!s.ok()) {
      return s;
    }

    std::string_view remaining = index_entries_view;
    while (!remaining.empty()) {
      IndexEntry entry;
      DecodeStatus ds = DecodeIndexEntry(&remaining, &entry);
      if (ds != DecodeStatus::kOk) {
        return ToStatus(ds, "sstable index entry");
      }
      reader->index_.push_back(std::move(entry));
    }
  }

  *out = std::move(reader);
  return Status::OK();
}

Status Reader::ReadBlock(const BlockHandle& handle, std::string* raw_block,
                          std::string_view* entries) const {
  // Every ReadBlock() call -- the index block at Open() and every data
  // block a Get()/ForEachEntry() reads via an index entry's handle --
  // goes through this same bounds check before touching the file, so an
  // out-of-range handle (footer-level or per-index-entry) is always
  // reported as Status::Corruption instead of reaching
  // util::PosixFile::ReadAt's up-front `size`-byte allocation. See
  // ValidateHandleInBounds and ADR 0011.
  Status bounds_status = ValidateHandleInBounds(handle, file_size_, "sstable block");
  if (!bounds_status.ok()) {
    return bounds_status;
  }

  bool short_read = false;
  Status s = file_->ReadAt(handle.offset, handle.size, raw_block, &short_read);
  if (!s.ok()) {
    return s;
  }
  if (short_read) {
    return Status::Corruption("short read while reading sstable block");
  }

  DecodeStatus ds = VerifyAndStripBlockTrailer(*raw_block, entries);
  if (ds != DecodeStatus::kOk) {
    return ToStatus(ds, "sstable block");
  }
  return Status::OK();
}

Status Reader::Get(Slice key, LookupResult* result, std::string* value,
                    std::uint64_t* sequence) const {
  auto it = std::lower_bound(
      index_.begin(), index_.end(), key,
      [](const IndexEntry& entry, Slice target) { return Slice(entry.last_key) < target; });
  if (it == index_.end()) {
    *result = LookupResult::kNotFound;
    return Status::OK();
  }

  std::string raw_block;
  std::string_view entries_view;
  Status s = ReadBlock(it->handle, &raw_block, &entries_view);
  if (!s.ok()) {
    return s;
  }

  while (!entries_view.empty()) {
    Entry entry;
    DecodeStatus ds = DecodeEntry(&entries_view, &entry);
    if (ds != DecodeStatus::kOk) {
      return ToStatus(ds, "sstable data entry");
    }

    if (Slice(entry.key) == key) {
      if (sequence != nullptr) {
        *sequence = entry.sequence;
      }
      if (entry.type == EntryType::kTombstone) {
        *result = LookupResult::kDeleted;
      } else {
        *value = std::move(entry.value);
        *result = LookupResult::kFound;
      }
      return Status::OK();
    }
    if (Slice(entry.key) > key) {
      break;  // sorted block: we have passed where `key` would be
    }
  }
  *result = LookupResult::kNotFound;
  return Status::OK();
}

Status Reader::ForEachEntry(const EntryCallback& callback) const {
  for (const IndexEntry& index_entry : index_) {
    std::string raw_block;
    std::string_view entries_view;
    Status s = ReadBlock(index_entry.handle, &raw_block, &entries_view);
    if (!s.ok()) {
      return s;
    }

    while (!entries_view.empty()) {
      Entry entry;
      DecodeStatus ds = DecodeEntry(&entries_view, &entry);
      if (ds != DecodeStatus::kOk) {
        return ToStatus(ds, "sstable data entry");
      }
      callback(entry);
    }
  }
  return Status::OK();
}

}  // namespace pebbledb::sstable
