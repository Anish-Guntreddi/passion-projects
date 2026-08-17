#include "pebbledb/status.hpp"

namespace pebbledb {

namespace {
const char* CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kNotFound:
      return "NotFound";
    case StatusCode::kCorruption:
      return "Corruption";
    case StatusCode::kIOError:
      return "IOError";
    case StatusCode::kInvalidArgument:
      return "InvalidArgument";
    case StatusCode::kNotSupported:
      return "NotSupported";
  }
  return "Unknown";
}
}  // namespace

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  std::string result = CodeName(code_);
  if (!message_.empty()) {
    result += ": ";
    result += message_;
  }
  return result;
}

}  // namespace pebbledb
