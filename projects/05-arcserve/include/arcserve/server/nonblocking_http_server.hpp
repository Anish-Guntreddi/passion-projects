#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "arcserve/net/socket.hpp"
#include "arcserve/reactor/connection.hpp"
#include "arcserve/reactor/epoll_reactor.hpp"
#include "arcserve/server/http_connection.hpp"
#include "arcserve/server/request_handler.hpp"

namespace arcserve::server {

// Phase 3: the same HTTP/1.1 subset (docs/protocol-scope.md) and the same
// HttpRequestParser Phase 1's BlockingHttpServer uses, served by a
// single-threaded epoll reactor instead of one-blocking-connection-at-a-
// time — many connections make progress concurrently on one thread, none
// of them ever blocking that thread on socket I/O (FR1). Handlers
// (RequestHandler) still run synchronously inline on the reactor thread,
// exactly as in Phase 1 — a worker pool for CPU-bound handler work is
// Phase 5's concern, not this one's.
//
// Concurrency/lifetime invariants:
//   - run() blocks the calling thread until stop() is observed; stop() is
//     thread-safe and wakes a blocked run() promptly (EpollReactor's
//     eventfd wakeup — no bounded-poll-interval compromise, unlike Phase
//     1's BlockingHttpServer, which had no such wakeup available).
//   - Every accepted client fd is owned by exactly one HttpConnection (via
//     its reactor::Connection base), itself owned by exactly one entry in
//     connections_, for the connection's entire lifetime. close_connection()
//     is the single teardown path (peer closed, I/O error, EPOLLHUP/
//     EPOLLERR, a parse error, "Connection: close", or the keep-alive
//     request-count bound) and the only place a client fd is ever removed
//     from the reactor/registry — no fd is ever leaked or double-closed.
//   - Partial writes: a response that doesn't fit in one send(2) call is
//     buffered in HttpConnection::base.write_buffer and flushed on
//     subsequent EPOLLOUT events (reactor::ConnectionState::kWriting /
//     kClosing) — see docs/decisions/0005-reactor-connection-model.md for
//     why this single-buffer approach is intentionally narrower than
//     Phase 4's planned write-queue/backpressure formalization.
//   - Not copyable or movable, matching BlockingHttpServer.
class NonblockingHttpServer {
 public:
  struct Config {
    std::uint16_t port = 0;  // 0 = kernel-assigned ephemeral port
    int backlog = 128;
    std::size_t read_chunk_bytes = 8192;
    // Same rationale as BlockingHttpServer::Config::max_keep_alive_requests:
    // an unbounded keep-alive loop on one connection would be a liveness
    // bug for every other connection multiplexed on this same reactor
    // thread.
    std::size_t max_keep_alive_requests = 1000;
  };

  NonblockingHttpServer(Config config, RequestHandler handler);

  NonblockingHttpServer(const NonblockingHttpServer&) = delete;
  NonblockingHttpServer& operator=(const NonblockingHttpServer&) = delete;
  NonblockingHttpServer(NonblockingHttpServer&&) = delete;
  NonblockingHttpServer& operator=(NonblockingHttpServer&&) = delete;
  ~NonblockingHttpServer() = default;

  // Blocks the calling thread, driving the reactor, until stop() is called.
  void run();

  // Thread-safe; see class-level invariants above.
  void stop() noexcept;

  // The actual bound port (resolves the ephemeral-port case). Valid
  // immediately after construction, before run() is ever called.
  [[nodiscard]] std::uint16_t port() const noexcept { return listener_.port(); }

  // Informal counter for tests; Phase 7 formalizes real observability
  // (FR8). Atomic because tests read it from a thread other than the one
  // running run(); only ever written from the reactor thread.
  [[nodiscard]] std::size_t connections_accepted() const noexcept {
    return connections_accepted_.load(std::memory_order_relaxed);
  }

 private:
  enum class FlushResult { kFlushed, kPending, kFailed };

  void on_listener_readable(int fd, std::uint32_t events);
  void on_client_event(int fd, std::uint32_t events);
  void process_readable(HttpConnection& conn);
  void drive_parser(HttpConnection& conn);
  FlushResult try_flush(reactor::Connection& conn);
  void close_connection(int fd) noexcept;

  net::TcpListener listener_;
  reactor::EpollReactor reactor_;
  std::unordered_map<int, std::unique_ptr<HttpConnection>> connections_;
  RequestHandler handler_;
  Config config_;
  std::atomic<std::size_t> connections_accepted_{0};
};

}  // namespace arcserve::server
