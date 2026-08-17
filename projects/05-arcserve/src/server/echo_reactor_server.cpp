#include "arcserve/server/echo_reactor_server.hpp"

#include <sys/epoll.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace arcserve::server {

EchoReactorServer::EchoReactorServer(Config config)
    : listener_(net::TcpListener::listen(config.port, config.backlog, /*nonblocking=*/true)),
      config_(config) {
  reactor_.add(listener_.native_handle(), reactor::kReadable,
               [this](int fd, std::uint32_t events) { on_listener_readable(fd, events); });
}

void EchoReactorServer::run() { reactor_.run(); }

void EchoReactorServer::stop() noexcept { reactor_.stop(); }

void EchoReactorServer::on_listener_readable(int /*fd*/, std::uint32_t /*events*/) {
  // Level-triggered epoll only guarantees "readable" fires again if the
  // backlog is still non-empty after this call — draining it in a loop
  // here (rather than accepting exactly one connection per event) means
  // one epoll_wait() wakeup is enough to absorb an accept burst instead of
  // needing one extra wakeup per queued connection.
  for (;;) {
    std::optional<net::FileDescriptor> client = listener_.accept_nonblocking();
    if (!client.has_value()) {
      return;
    }

    int client_fd = client->get();
    std::string peer = net::describe_peer(client_fd);
    auto connection = std::make_unique<reactor::Connection>(std::move(*client), std::move(peer));
    connections_.emplace(client_fd, std::move(connection));
    connections_accepted_.fetch_add(1, std::memory_order_relaxed);

    reactor_.add(client_fd, reactor::kReadable,
                 [this](int fd, std::uint32_t events) { on_client_event(fd, events); });
  }
}

void EchoReactorServer::on_client_event(int fd, std::uint32_t events) {
  auto it = connections_.find(fd);
  if (it == connections_.end()) {
    return;  // Defensive: see EpollReactor::dispatch_ready's comment.
  }
  reactor::Connection& conn = *it->second;

  if ((events & (EPOLLERR | EPOLLHUP)) != 0) {
    close_connection(fd);
    return;
  }

  if ((events & reactor::kWritable) != 0) {
    FlushResult result = try_flush(conn);
    if (result == FlushResult::kFailed) {
      close_connection(fd);
      return;
    }
    if (result == FlushResult::kFlushed) {
      conn.set_state(reactor::ConnectionState::kReading);
      reactor_.modify(fd, reactor::kReadable);
    }
    // kPending: stay in kWriting with EPOLLOUT still armed.
  }

  // Only relevant while kReading: interest is never kReadable at the same
  // time as kWriting/kClosing (see reactor::ConnectionState's docs), so
  // this check is really "did epoll actually report readability", but
  // spelling out the state guard keeps the invariant load-bearing rather
  // than implicit.
  if ((events & reactor::kReadable) != 0 && conn.state() == reactor::ConnectionState::kReading) {
    process_readable(conn);
  }
}

void EchoReactorServer::process_readable(reactor::Connection& conn) {
  std::vector<char> buffer(config_.read_chunk_bytes);
  for (;;) {
    net::IoResult result = net::read_some(conn.fd(), buffer.data(), buffer.size());
    if (result.bytes_transferred > 0) {
      conn.read_buffer.append(buffer.data(), result.bytes_transferred);
      conn.touch();
    }
    if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
      close_connection(conn.fd());
      return;
    }
    if (result.status == net::IoStatus::WouldBlock || result.bytes_transferred < buffer.size()) {
      // WouldBlock: the socket is definitively drained. A short read that
      // wasn't WouldBlock is not a hard guarantee more isn't available
      // immediately, but level-triggered epoll re-reports readability on
      // the next epoll_wait() if so — stopping here trades one possible
      // extra wakeup for one fewer recv(2) call in the common case.
      break;
    }
  }

  if (conn.read_buffer.empty()) {
    return;  // Nothing new arrived (shouldn't happen under EPOLLIN, but
             // harmless if it does).
  }

  conn.write_buffer += conn.read_buffer;
  conn.read_buffer.clear();

  FlushResult flush_result = try_flush(conn);
  if (flush_result == FlushResult::kFailed) {
    close_connection(conn.fd());
    return;
  }
  if (flush_result == FlushResult::kPending) {
    // Pause reading until the echo fully drains — see
    // docs/decisions/0005-reactor-connection-model.md decision 3 for why.
    conn.set_state(reactor::ConnectionState::kWriting);
    reactor_.modify(conn.fd(), reactor::kWritable);
  }
}

EchoReactorServer::FlushResult EchoReactorServer::try_flush(reactor::Connection& conn) {
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

void EchoReactorServer::close_connection(int fd) noexcept {
  reactor_.remove(fd);
  connections_.erase(fd);
}

}  // namespace arcserve::server
