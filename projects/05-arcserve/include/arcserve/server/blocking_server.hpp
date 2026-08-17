#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "arcserve/net/file_descriptor.hpp"
#include "arcserve/net/socket.hpp"
#include "arcserve/protocol/http_message.hpp"

namespace arcserve::server {

// Given a fully-parsed request, produce the response to send. Invoked
// synchronously on the connection-handling call stack — there is no worker
// pool yet (that's Phase 5); a slow handler blocks only the connection
// currently being served, since connections are handled one at a time to
// completion (see BlockingHttpServer below).
using RequestHandler = std::function<protocol::HttpResponse(const protocol::HttpRequest&)>;

// A deliberately single-threaded, blocking, one-connection-at-a-time
// TCP/HTTP server. Its purpose is to validate protocol and parser
// correctness (FR3) in isolation from the nonblocking epoll reactor, which
// Phase 2 introduces — this is exactly the Phase 1 roadmap scope ("Minimal
// blocking echo/framed server validating protocol/parser semantics").
//
// Concurrency/lifetime invariants:
//   - serve() runs on whichever thread calls it. It never returns control
//     to the caller until stop() is observed (or an unrecoverable poll()
//     failure occurs) — callers that need to do anything else concurrently
//     (as the integration tests do) must run it on its own thread.
//   - stop() is safe to call from any thread, including concurrently with
//     serve() running on another thread; it only ever writes a
//     std::atomic<bool>. serve()'s accept-poll loop observes it within
//     poll_interval_ms (bounded — see Config), so shutdown is prompt even
//     though there is no reactor here yet to interrupt via an eventfd/
//     self-pipe wakeup (that mechanism arrives with the epoll reactor in
//     Phase 2; a plain bounded poll() is sufficient when there is only one
//     fd — the listener — to ever wait on).
//   - stop() does not abort a connection already being served mid-request;
//     serve() finishes handling the in-flight connection (or its
//     keep-alive loop reaches Connection: close / EOF / its own
//     max_keep_alive_requests bound) before the flag is checked again.
//     There is no forced mid-request abort in Phase 1 (timeouts and
//     overload-driven forced closes are Phase 6 concerns).
//   - The listening socket is owned exclusively by this object (via
//     net::TcpListener, itself built on net::FileDescriptor); no other code
//     may close it. Each accepted client fd is owned locally by
//     handle_connection() for the connection's lifetime and is closed via
//     its FileDescriptor destructor on every return path (normal
//     completion, error, or peer disconnect) — no fd is ever leaked or
//     double-closed.
//   - Not copyable or movable: a BlockingHttpServer's identity (its bound
//     listening socket, its stop flag) must not be duplicated or relocated
//     while serve() may be running against it.
class BlockingHttpServer {
 public:
  struct Config {
    std::uint16_t port = 0;  // 0 = kernel-assigned ephemeral port
    int backlog = 128;
    int poll_interval_ms = 100;  // upper bound on stop() latency
    // Bounded so no single connection can hold the (single-threaded)
    // server hostage forever via keep-alive, even if every request is
    // otherwise well-formed and fast — an unbounded loop here would be a
    // liveness bug for every *other* client waiting in the accept backlog.
    std::size_t max_keep_alive_requests = 1000;
    std::size_t read_chunk_bytes = 8192;
  };

  BlockingHttpServer(Config config, RequestHandler handler);

  BlockingHttpServer(const BlockingHttpServer&) = delete;
  BlockingHttpServer& operator=(const BlockingHttpServer&) = delete;
  BlockingHttpServer(BlockingHttpServer&&) = delete;
  BlockingHttpServer& operator=(BlockingHttpServer&&) = delete;
  ~BlockingHttpServer() = default;

  // Blocks the calling thread, accepting and serving connections
  // sequentially, until stop() is called.
  void serve();

  // Thread-safe; see class-level invariants above.
  void stop() noexcept;

  // The actual bound port (resolves the ephemeral-port case). Valid
  // immediately after construction, before serve() is ever called.
  [[nodiscard]] std::uint16_t port() const noexcept { return listener_.port(); }

 private:
  void handle_connection(net::FileDescriptor client);

  net::TcpListener listener_;
  RequestHandler handler_;
  Config config_;
  std::atomic<bool> stop_requested_{false};
};

// stop() is called from main.cpp's signal handler, which may only touch
// plain lock-free atomic operations ([support.signal]) — this makes that
// requirement a compile-time guarantee instead of a platform assumption.
static_assert(std::atomic<bool>::is_always_lock_free);

}  // namespace arcserve::server
