#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "pebbledb/status.hpp"
#include "pebbledb/util/posix_file.hpp"
#include "pebbledb/wal/record.hpp"

namespace pebbledb::wal {

// Recovery policy (docs/durability.md): a WAL is read front-to-back and
// replay stops at the first record that is not fully present and valid.
// Everything before that point is replayed exactly; nothing after it is
// trusted, even if later bytes happen to look like a well-formed record
// (PebbleDB does not attempt to resynchronize past a bad record — see
// ADR 0006 for why that is the right tradeoff for an append-only,
// single-writer WAL).
enum class ReadResult {
  kOk,                 // a valid record was decoded into the output Record
  kEndOfLog,           // clean EOF, exactly at a record boundary
  kTruncated,          // a record was started but not fully present
                       // (header or payload torn) — the expected shape of
                       // an unclean shutdown's last, in-flight write
  kCorruption,         // a full record's bytes were present but failed
                       // validation (bad magic, bad length, bad op byte,
                       // or checksum mismatch)
  kUnsupportedVersion, // a full header was present with a format_version
                       // newer than this reader understands
};

const char* ReadResultName(ReadResult result);

class Reader {
 public:
  static Status Open(const std::string& path, std::unique_ptr<Reader>* out);

  ~Reader() = default;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  // Reads and decodes the next record into `*record`. Once this returns
  // anything other than kOk, the reader is exhausted: every subsequent
  // call returns that same terminal result again without reading any
  // more bytes (see the recovery policy above).
  ReadResult ReadNext(Record* record);

  struct ReplayResult {
    std::uint64_t records_replayed = 0;
    ReadResult terminal = ReadResult::kEndOfLog;
  };

  // Calls ReadNext repeatedly, invoking `on_record` for each kOk record
  // in order, until a non-kOk result is reached.
  ReplayResult Replay(const std::function<void(const Record&)>& on_record);

 private:
  explicit Reader(std::unique_ptr<util::PosixFile> file) : file_(std::move(file)) {}

  std::unique_ptr<util::PosixFile> file_;
  bool exhausted_ = false;
  ReadResult terminal_result_ = ReadResult::kEndOfLog;
};

}  // namespace pebbledb::wal
