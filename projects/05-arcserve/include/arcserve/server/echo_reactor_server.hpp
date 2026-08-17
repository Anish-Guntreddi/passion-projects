#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "arcserve/net/socket.hpp"
#include "arcserve/reactor/connection.hpp"
#include "arcserve/reactor/epoll_reactor.hpp"

namespace arcserve::server {

// A minimal nonblocking TCP echo server: whatever bytes a client sends are
// written back verbatim, in order. Its sole purpose is to prove out the
// Phase 2 roadmap deliverable ("Reactor + connection registry; echo under
// many connections") in isolation from HTTP parsing — this class does not
// include arcserve/protocol/anything, deliberately: it is the concrete
// evidence, referenced by docs/decisions/0005-reactor-connection-model.md,
// that reactor::EpollReactor + reactor::Connection are genuinely
// protocol-agnostic infrastructure rather than HTTP machinery in disguise.
// See NonblockingHttpServer (Phase 3) for the protocol-aware server built
// on these same primitives.
//
// Concurrency/lifetime invariants:
//   - run() blocks the calling thread until stop() is observed; like
//     BlockingHttpServer, callers needing concurrent access (tests) run it
//     on its own thread.
//   - stop() is thread-safe (forwards to EpollReactor::stop(), itself
//     documented thread-safe) and wakes a blocked run() promptly.
//   - Every accepted client fd is owned by exactly one reactor::Connection,
//     itself owned by exactly one entry in connections_, for the
//     connection's entire lifetime. close_connection() is the single path
//     that erases it (and, via ~Connection, closes the fd) — called from
//     every place a connection ends (peer closed, I/O error, EPOLLHUP/
//     EPOLLERR). No fd is ever leaked or double-closed.
//   - Not copyable or movable, matching BlockingHttpServer.
class EchoReactorServer {
 public:
  struct Config {
    std::uint16_t port = 0;  // 0 = kernel-assigned ephemeral port
    int backlog = 128;
    std::size_t read_chunk_bytes = 8192;
  };

  explicit EchoReactorServer(Config config);

  EchoReactorServer(const EchoReactorServer&) = delete;
  EchoReactorServer& operator=(const EchoReactorServer&) = delete;
  EchoReactorServer(EchoReactorServer&&) = delete;
  EchoReactorServer& operator=(EchoReactorServer&&) = delete;
  ~EchoReactorServer() = default;

  // Blocks the calling thread, driving the reactor, until stop() is called.
  void run();

  // Thread-safe; see class-level invariants above.
  void stop() noexcept;

  // The actual bound port (resolves the ephemeral-port case). Valid
  // immediately after construction, before run() is ever called.
  [[nodiscard]] std::uint16_t port() const noexcept { return listener_.port(); }

  // Informal counters for tests; Phase 7 formalizes real observability
  // (structured logging + metrics counters, FR8). connections_accepted()
  // is atomic because tests read it from a thread other than the one
  // running run(); it is only ever written from the reactor thread.
  [[nodiscard]] std::size_t connections_accepted() const noexcept {
    return connections_accepted_.load(std::memory_order_relaxed);
  }

 private:
  enum class FlushResult { kFlushed, kPending, kFailed };

  void on_listener_readable(int fd, std::uint32_t events);
  void on_client_event(int fd, std::uint32_t events);
  void process_readable(reactor::Connection& conn);
  FlushResult try_flush(reactor::Connection& conn);
  void close_connection(int fd) noexcept;

  net::TcpListener listener_;
  reactor::EpollReactor reactor_;
  std::unordered_map<int, std::unique_ptr<reactor::Connection>> connections_;
  Config config_;
  std::atomic<std::size_t> connections_accepted_{0};
};

}  // namespace arcserve::server
