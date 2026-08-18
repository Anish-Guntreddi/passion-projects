// inspect_sstable: Phase 3's file-inspector deliverable (spec FR3, "file
// inspector tools"). Opens an SSTable file written by sstable::Writer and
// prints its footer, index (one entry per data block), and, with --dump,
// every entry in the table. Read-only: never writes to the file it
// inspects (matches invariant 2, "SSTable contents are immutable after
// successful creation" -- an inspector must not be a hole in that).
//
// Usage: inspect_sstable <path> [--dump]

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "pebbledb/entry_type.hpp"
#include "pebbledb/sstable/format.hpp"
#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/status.hpp"

namespace {

// Keys/values are binary-safe (spec §1.2) and may contain unprintable or
// non-UTF8 bytes; render them as a mix of printable ASCII and \xHH escapes
// so the tool's output is always safe to print to a terminal.
std::string EscapeForDisplay(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  char hex[5];
  for (unsigned char c : s) {
    if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\') {
      out.push_back(static_cast<char>(c));
    } else {
      std::snprintf(hex, sizeof(hex), "\\x%02x", c);
      out += hex;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: inspect_sstable <path> [--dump]\n";
    return 2;
  }
  const std::string path = argv[1];
  const bool dump = (argc >= 3 && std::string(argv[2]) == "--dump");

  std::unique_ptr<pebbledb::sstable::Reader> reader;
  pebbledb::Status s = pebbledb::sstable::Reader::Open(path, &reader);
  if (!s.ok()) {
    std::cerr << "failed to open '" << path << "': " << s.ToString() << "\n";
    return 1;
  }

  const pebbledb::sstable::Footer& footer = reader->footer();
  std::cout << "sstable: " << path << "\n";
  std::cout << "  format_version: " << static_cast<int>(footer.format_version) << "\n";
  std::cout << "  num_entries: " << footer.num_entries << "\n";
  std::cout << "  index_block: offset=" << footer.index_handle.offset
             << " size=" << footer.index_handle.size << "\n";
  std::cout << "  filter_block: offset=" << footer.filter_handle.offset
             << " size=" << footer.filter_handle.size;
  if (footer.filter_handle.size == 0) {
    std::cout << " (absent -- filter disabled for this table, or a pre-Phase-5 file)";
  } else if (footer.filter_handle.size > pebbledb::sstable::kBlockTrailerSize + 1) {
    // content = block bytes minus the trailer (kBlockTrailerSize); the
    // content's first byte is num_probes, the rest is the bit array --
    // see bloom/bloom_filter.hpp.
    const std::uint64_t content_size =
        footer.filter_handle.size - pebbledb::sstable::kBlockTrailerSize;
    std::cout << " (Bloom filter, " << (content_size - 1) * 8 << " bits)";
  } else {
    std::cout << " (malformed -- too small to be a valid filter block)";
  }
  std::cout << "\n";
  std::cout << "  data_blocks: " << reader->index_entry_count() << "\n";

  std::size_t block_no = 0;
  for (const auto& entry : reader->index_entries()) {
    std::cout << "    [" << block_no++ << "] last_key=\"" << EscapeForDisplay(entry.last_key)
               << "\" offset=" << entry.handle.offset << " size=" << entry.handle.size << "\n";
  }

  if (dump) {
    std::cout << "  entries:\n";
    s = reader->ForEachEntry([](const pebbledb::sstable::Entry& e) {
      std::cout << "    key=\"" << EscapeForDisplay(e.key) << "\" seq=" << e.sequence << " type="
                 << (e.type == pebbledb::EntryType::kTombstone ? "tombstone" : "value");
      if (e.type != pebbledb::EntryType::kTombstone) {
        std::cout << " value=\"" << EscapeForDisplay(e.value) << "\"";
      }
      std::cout << "\n";
    });
    if (!s.ok()) {
      std::cerr << "error while dumping entries: " << s.ToString() << "\n";
      return 1;
    }
  }

  return 0;
}
