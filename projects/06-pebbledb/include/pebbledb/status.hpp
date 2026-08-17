#pragma once

#include <cstdint>
#include <string>

namespace pebbledb {

// PebbleDB's error model (Phase 0 deliverable, spec Part 1 §1.2; see
// docs/decisions/0002-status-and-byte-ownership.md). Every fallible
// operation in the public API returns a Status by value instead of
// throwing: storage engines fail routinely (missing key, truncated
// record, disk full) and callers up and down the write/read path need to
// branch on *which* failure occurred, which a single unstructured
// exception type does not make convenient. A default-constructed Status
// is OK, matching the common "the happy path costs nothing to spell" idiom.
enum class StatusCode : std::uint8_t {
  kOk = 0,
  kNotFound,
  kCorruption,
  kIOError,
  kInvalidArgument,
  kNotSupported,
};

class Status {
 public:
  // OK by default.
  Status() = default;

  static Status OK() { return Status(); }
  static Status NotFound(std::string message = {}) {
    return Status(StatusCode::kNotFound, std::move(message));
  }
  static Status Corruption(std::string message) {
    return Status(StatusCode::kCorruption, std::move(message));
  }
  static Status IOError(std::string message) {
    return Status(StatusCode::kIOError, std::move(message));
  }
  static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  static Status NotSupported(std::string message = {}) {
    return Status(StatusCode::kNotSupported, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  bool IsNotFound() const { return code_ == StatusCode::kNotFound; }
  bool IsCorruption() const { return code_ == StatusCode::kCorruption; }
  bool IsIOError() const { return code_ == StatusCode::kIOError; }
  bool IsInvalidArgument() const { return code_ == StatusCode::kInvalidArgument; }
  bool IsNotSupported() const { return code_ == StatusCode::kNotSupported; }

  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  // "OK" for the OK status; otherwise "<Kind>: <message>" (or just
  // "<Kind>" if the message is empty).
  std::string ToString() const;

 private:
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}  // namespace pebbledb
