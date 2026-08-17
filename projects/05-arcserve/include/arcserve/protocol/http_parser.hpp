#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "arcserve/protocol/http_message.hpp"

namespace arcserve::protocol {

// Outcome of a single feed() call.
enum class ParseStatus {
  kNeedMoreData,  // Not yet a complete request; call feed() again with more bytes.
  kComplete,      // request() now holds a fully-parsed HttpRequest.
  kError,         // Malformed/oversized input; error_response() holds the
                   // response to send before closing the connection.
};

// Handwritten, single-pass, byte-incremental parser for the HTTP/1.1
// request subset in docs/protocol-scope.md (rationale for hand-rolling it
// instead of a library: docs/decisions/0003-handwritten-http-parser.md).
//
// Usage: construct one instance per connection. Call feed() with each
// chunk of bytes read off the socket — any chunk size, including single
// bytes, since the parser keeps all cross-call state internally, so
// fragmented TCP delivery (FR3) is handled transparently. On kComplete,
// `data` may still contain bytes belonging to the *next* pipelined
// request (feed() only consumes what one request needs); read request(),
// then call reset() before feeding the parser again to parse that next
// request on the same keep-alive connection.
//
// Not thread-safe; one instance belongs to exactly one connection, used
// from exactly one thread at a time.
class HttpRequestParser {
 public:
  HttpRequestParser() = default;

  // Consumes a prefix of `data` (removed via data.remove_prefix as bytes
  // are parsed) and returns how far parsing got. On kNeedMoreData, all of
  // `data` was consumed into internal buffers. On kComplete or kError,
  // `data` is left pointing at whatever bytes (if any) were not needed to
  // reach that outcome.
  ParseStatus feed(std::string_view& data);

  // Valid after feed() returns kComplete.
  [[nodiscard]] HttpRequest& request() noexcept { return request_; }
  [[nodiscard]] const HttpRequest& request() const noexcept { return request_; }

  // Valid after feed() returns kError: the response the server should send
  // (and then close the connection) for this malformed/oversized request.
  [[nodiscard]] const HttpResponse& error_response() const noexcept { return error_response_; }

  // Returns this parser to its initial state so the same instance can
  // parse the next request on a keep-alive connection. Required after both
  // kComplete and kError before calling feed() again.
  void reset();

  // Hard caps enforced by this parser; see docs/protocol-scope.md for the
  // rationale for each value. Oversized/unsupported input is rejected
  // predictably (FR3) rather than allowed to grow buffers without bound.
  static constexpr std::size_t kMaxRequestLineLength = 8192;
  static constexpr std::size_t kMaxHeaderBytes = 8192;
  static constexpr std::size_t kMaxHeaderCount = 100;
  static constexpr std::size_t kMaxBodyBytes = 1 << 20;  // 1 MiB

 private:
  enum class State { kRequestLine, kHeaders, kBody, kDone };

  // Internal step outcome, distinct from the public ParseStatus: kAdvanced
  // means "made progress, state_ may have changed, caller should look at
  // state_ to decide what happens next" — it is not exposed to callers of
  // feed(), which only ever returns kNeedMoreData / kComplete / kError.
  enum class StepResult { kNeedMoreData, kAdvanced, kFailed };

  StepResult parse_request_line(std::string_view& data);
  StepResult parse_headers(std::string_view& data);
  StepResult parse_body(std::string_view& data);

  StepResult finish_request_line(std::string_view line);
  StepResult consume_header_line(std::string_view line);
  StepResult finish_headers();

  StepResult fail_step(int status_code, std::string_view reason_phrase, std::string_view body);

  State state_ = State::kRequestLine;
  std::string line_buffer_;  // accumulates a partial request-line/header line across feed() calls
  std::size_t header_bytes_seen_ = 0;
  std::optional<std::size_t> content_length_;
  HttpRequest request_;
  HttpResponse error_response_;
};

}  // namespace arcserve::protocol
