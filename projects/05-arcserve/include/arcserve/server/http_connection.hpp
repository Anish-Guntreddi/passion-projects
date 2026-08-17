#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "arcserve/net/file_descriptor.hpp"
#include "arcserve/protocol/http_parser.hpp"
#include "arcserve/reactor/connection.hpp"

namespace arcserve::server {

// Per-connection state for NonblockingHttpServer (Phase 3): composes the
// protocol-agnostic reactor::Connection (Phase 2 — fd ownership, I/O
// buffers, lifecycle state, peer metadata, activity timestamp) with the
// HTTP-specific state FR2 also calls for ("parser state"): one
// HttpRequestParser per connection — parser state is inherently
// per-connection, since it accumulates a partial request line/headers/body
// across however many reactor read events it takes to arrive — plus the
// keep-alive request counter that bounds a single connection exactly like
// BlockingHttpServer::Config::max_keep_alive_requests does in Phase 1.
//
// Composition (a reactor::Connection member), not inheritance, by design:
// nothing here is ever accessed through a reactor::Connection base
// pointer/reference, so there is no slicing risk to guard against, and it
// keeps the HTTP-specific fields visibly distinct from the generic ones
// this class did not invent.
class HttpConnection {
 public:
  HttpConnection(net::FileDescriptor fd, std::string peer) noexcept
      : base(std::move(fd), std::move(peer)) {}

  HttpConnection(const HttpConnection&) = delete;
  HttpConnection& operator=(const HttpConnection&) = delete;
  HttpConnection(HttpConnection&&) = delete;  // same rationale as reactor::Connection
  HttpConnection& operator=(HttpConnection&&) = delete;
  ~HttpConnection() = default;

  reactor::Connection base;
  protocol::HttpRequestParser parser;
  std::size_t requests_served = 0;
};

}  // namespace arcserve::server
