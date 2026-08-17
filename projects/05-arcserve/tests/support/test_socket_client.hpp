#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "arcserve/net/file_descriptor.hpp"

namespace arcserve::testing {

// Minimal blocking TCP client used only by integration tests to script
// raw, byte-level exchanges against BlockingHttpServer — including
// deliberately fragmented writes that exercise the parser's cross-chunk
// state, which a higher-level HTTP client library would hide from us.
class RawTcpClient {
 public:
  explicit RawTcpClient(std::uint16_t port);

  RawTcpClient(const RawTcpClient&) = delete;
  RawTcpClient& operator=(const RawTcpClient&) = delete;
  RawTcpClient(RawTcpClient&&) = delete;
  RawTcpClient& operator=(RawTcpClient&&) = delete;
  ~RawTcpClient() = default;

  // Sends `data` as one or more send(2) calls of at most `chunk_size`
  // bytes each (0 = the whole buffer in one call), to simulate fragmented
  // delivery. Returns false on send failure.
  bool send_fragmented(std::string_view data, std::size_t chunk_size = 0);

  // A single blocking read: one recv(2) call, up to max_bytes. Returns ""
  // on a clean close or a genuine zero-byte read; check is_closed() to
  // tell those apart.
  std::string read_chunk(std::size_t max_bytes = 4096);

  [[nodiscard]] bool is_closed() const noexcept { return closed_; }

  // Closes the socket immediately, simulating an abrupt client disconnect.
  void close_now();

 private:
  arcserve::net::FileDescriptor fd_;
  bool closed_ = false;
};

// Reads one full HTTP response (status line + headers + body) from
// `client`, using the server's own framing guarantee: every response
// BlockingHttpServer sends always carries a Content-Length header (see
// HttpResponse::serialize()). Blocks until the full response arrives or
// the connection closes early, in which case whatever partial bytes were
// read are returned as-is.
std::string read_one_http_response(RawTcpClient& client);

}  // namespace arcserve::testing
