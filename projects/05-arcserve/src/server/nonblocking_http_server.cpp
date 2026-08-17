#include "arcserve/server/nonblocking_http_server.hpp"

#include <sys/epoll.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arcserve::server {

NonblockingHttpServer::NonblockingHttpServer(Config config, RequestHandler handler)
    : listener_(net::TcpListener::listen(config.port, config.backlog, /*nonblocking=*/true)),
      handler_(std::move(handler)),
      config_(config) {
  reactor_.add(listener_.native_handle(), reactor::kReadable,
               [this](int fd, std::uint32_t events) { on_listener_readable(fd, events); });
}

void NonblockingHttpServer::run() { reactor_.run(); }

void NonblockingHttpServer::stop() noexcept { reactor_.stop(); }

void NonblockingHttpServer::on_listener_readable(int /*fd*/, std::uint32_t /*events*/) {
  for (;;) {
    std::optional<net::FileDescriptor> client = listener_.accept_nonblocking();
    if (!client.has_value()) {
      return;
    }

    int client_fd = client->get();
    std::string peer = net::describe_peer(client_fd);
    auto connection = std::make_unique<HttpConnection>(std::move(*client), std::move(peer));
    connections_.emplace(client_fd, std::move(connection));
    connections_accepted_.fetch_add(1, std::memory_order_relaxed);

    reactor_.add(client_fd, reactor::kReadable,
                 [this](int fd, std::uint32_t events) { on_client_event(fd, events); });
  }
}

void NonblockingHttpServer::on_client_event(int fd, std::uint32_t events) {
  auto it = connections_.find(fd);
  if (it == connections_.end()) {
    return;  // Defensive: see EpollReactor::dispatch_ready's comment.
  }
  HttpConnection& conn = *it->second;

  if ((events & (EPOLLERR | EPOLLHUP)) != 0) {
    close_connection(fd);
    return;
  }

  if ((events & reactor::kWritable) != 0) {
    FlushResult result = try_flush(conn.base);
    if (result == FlushResult::kFailed) {
      close_connection(fd);
      return;
    }
    if (result == FlushResult::kFlushed) {
      if (conn.base.state() == reactor::ConnectionState::kClosing) {
        close_connection(fd);
        return;
      }
      conn.base.set_state(reactor::ConnectionState::kReading);
      reactor_.modify(fd, reactor::kReadable);
    }
    // kPending: stay in kWriting/kClosing with EPOLLOUT still armed.
  }

  // Interest is never kReadable at the same time as kWriting/kClosing (see
  // reactor::ConnectionState's docs), so this state guard is really just
  // making that invariant load-bearing rather than implicit.
  if ((events & reactor::kReadable) != 0 &&
      conn.base.state() == reactor::ConnectionState::kReading) {
    process_readable(conn);
  }
}

void NonblockingHttpServer::process_readable(HttpConnection& conn) {
  std::vector<char> buffer(config_.read_chunk_bytes);
  for (;;) {
    net::IoResult result = net::read_some(conn.base.fd(), buffer.data(), buffer.size());
    if (result.bytes_transferred > 0) {
      conn.base.read_buffer.append(buffer.data(), result.bytes_transferred);
      conn.base.touch();
    }
    if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
      close_connection(conn.base.fd());
      return;
    }
    if (result.status == net::IoStatus::WouldBlock || result.bytes_transferred < buffer.size()) {
      break;  // See EchoReactorServer::process_readable's identical comment.
    }
  }

  drive_parser(conn);
}

void NonblockingHttpServer::drive_parser(HttpConnection& conn) {
  std::string_view view(conn.base.read_buffer);
  bool close_after_flush = false;

  for (;;) {
    protocol::ParseStatus status = conn.parser.feed(view);

    if (status == protocol::ParseStatus::kNeedMoreData) {
      break;  // `view` was fully consumed into the parser's internal state;
               // wait for the next readable event.
    }

    if (status == protocol::ParseStatus::kError) {
      conn.base.write_buffer += conn.parser.error_response().serialize();
      close_after_flush = true;
      view = std::string_view();  // discard any trailing bytes; closing regardless
      break;
    }

    // kComplete.
    protocol::HttpResponse response = handler_(conn.parser.request());
    bool keep_alive = response.keep_alive && conn.parser.request().keep_alive();
    response.keep_alive = keep_alive;
    conn.base.write_buffer += response.serialize();
    ++conn.requests_served;
    conn.parser.reset();

    if (!keep_alive || conn.requests_served >= config_.max_keep_alive_requests) {
      close_after_flush = true;
      break;
    }
    // Otherwise loop again: `view` may already hold the start of the next
    // pipelined request, delivered in the same recv(2) as this one.
  }

  conn.base.read_buffer = std::string(view);

  FlushResult flush_result = try_flush(conn.base);
  if (flush_result == FlushResult::kFailed) {
    close_connection(conn.base.fd());
    return;
  }
  if (flush_result == FlushResult::kFlushed) {
    if (close_after_flush) {
      close_connection(conn.base.fd());
    }
    return;  // Otherwise: stay kReading, interest already kReadable.
  }

  // kPending: the response(s) didn't fully fit in one send(2) call.
  conn.base.set_state(close_after_flush ? reactor::ConnectionState::kClosing
                                         : reactor::ConnectionState::kWriting);
  reactor_.modify(conn.base.fd(), reactor::kWritable);
}

NonblockingHttpServer::FlushResult NonblockingHttpServer::try_flush(reactor::Connection& conn) {
  if (conn.write_buffer.empty()) {
    return FlushResult::kFlushed;
  }
  net::IoResult result =
      net::write_all(conn.fd(), conn.write_buffer.data(), conn.write_buffer.size());
  conn.write_buffer.erase(0, result.bytes_transferred);
  if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
    return FlushResult::kFailed;
  }
  if (result.bytes_transferred > 0) {
    conn.touch();
  }
  return conn.write_buffer.empty() ? FlushResult::kFlushed : FlushResult::kPending;
}

void NonblockingHttpServer::close_connection(int fd) noexcept {
  reactor_.remove(fd);
  connections_.erase(fd);
}

}  // namespace arcserve::server
