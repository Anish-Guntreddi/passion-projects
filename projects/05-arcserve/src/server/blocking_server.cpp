#include "arcserve/server/blocking_server.hpp"

#include <poll.h>

#include <cerrno>
#include <string>
#include <string_view>
#include <vector>

#include "arcserve/protocol/http_parser.hpp"

namespace arcserve::server {

BlockingHttpServer::BlockingHttpServer(Config config, RequestHandler handler)
    : listener_(net::TcpListener::listen(config.port, config.backlog)),
      handler_(std::move(handler)),
      config_(config) {}

void BlockingHttpServer::stop() noexcept { stop_requested_.store(true, std::memory_order_relaxed); }

void BlockingHttpServer::serve() {
  stop_requested_.store(false, std::memory_order_relaxed);

  while (!stop_requested_.load(std::memory_order_relaxed)) {
    pollfd pfd{};
    pfd.fd = listener_.native_handle();
    pfd.events = POLLIN;

    int ready = ::poll(&pfd, 1, config_.poll_interval_ms);
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;  // Unexpected poll() failure: stop serving rather than spin.
    }
    if (ready == 0) {
      continue;  // Timeout — recheck stop_requested_.
    }

    net::FileDescriptor client = listener_.accept();
    handle_connection(std::move(client));
  }
}

void BlockingHttpServer::handle_connection(net::FileDescriptor client) {
  protocol::HttpRequestParser parser;
  std::vector<char> read_buffer(config_.read_chunk_bytes);
  std::string leftover;  // bytes already read off the socket but not yet consumed by the parser
  std::size_t requests_served = 0;

  for (;;) {
    std::string_view view(leftover);
    protocol::ParseStatus status = parser.feed(view);
    // `view` now points at whatever suffix of `leftover` the parser didn't
    // need. Copy it into a fresh string rather than leftover.assign(view):
    // view aliases leftover's current buffer, and self-assigning a
    // std::string from a substring view of itself is not something the
    // standard guarantees is safe.
    leftover = std::string(view);

    if (status == protocol::ParseStatus::kNeedMoreData) {
      net::IoResult result = net::read_some(client.get(), read_buffer.data(), read_buffer.size());
      if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
        return;  // Peer gone (mid-request or between requests) or unrecoverable I/O error.
      }
      leftover.append(read_buffer.data(), result.bytes_transferred);
      continue;
    }

    if (status == protocol::ParseStatus::kError) {
      std::string wire = parser.error_response().serialize();
      // Best-effort: we're closing the connection regardless of whether
      // the client actually received the error response, so the result
      // isn't actionable here.
      (void)net::write_all(client.get(), wire.data(), wire.size());
      return;  // Protocol errors always close the connection.
    }

    // kComplete.
    protocol::HttpResponse response = handler_(parser.request());
    bool keep_alive = response.keep_alive && parser.request().keep_alive();
    response.keep_alive = keep_alive;

    std::string wire = response.serialize();
    net::IoResult write_result = net::write_all(client.get(), wire.data(), wire.size());
    if (write_result.status != net::IoStatus::Ok) {
      return;  // Client vanished mid-write (or a slow-client cutoff — Phase 4's concern).
    }

    ++requests_served;
    if (!keep_alive || requests_served >= config_.max_keep_alive_requests) {
      return;
    }

    parser.reset();
    // Loop to parse the next request. `leftover` may already hold the
    // start of a pipelined next request, or be empty (in which case
    // feed() immediately reports kNeedMoreData and we read more).
  }
}

}  // namespace arcserve::server
