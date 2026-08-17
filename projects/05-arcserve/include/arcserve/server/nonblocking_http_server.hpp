#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#include "arcserve/concurrency/bounded_work_queue.hpp"
#include "arcserve/concurrency/worker_pool.hpp"
#include "arcserve/net/file_descriptor.hpp"
#include "arcserve/net/socket.hpp"
#include "arcserve/protocol/http_message.hpp"
#include "arcserve/reactor/connection.hpp"
#include "arcserve/reactor/epoll_reactor.hpp"
#include "arcserve/server/http_connection.hpp"
#include "arcserve/server/request_handler.hpp"

namespace arcserve::server {

// Phase 3: the same HTTP/1.1 subset (docs/protocol-scope.md) and the same
// HttpRequestParser Phase 1's BlockingHttpServer uses, served by a
// single-threaded epoll reactor instead of one-blocking-connection-at-a-
// time — many connections make progress concurrently on one thread, none
// of them ever blocking that thread on socket I/O (FR1).
//
// Phase 5 adds an optional worker-pool dispatch mode (FR4): pass a
// concurrency::WorkerPool via Config::worker_pool and every parsed
// request's handler_() call runs on one of that pool's worker threads
// instead of inline on the reactor thread — see "Worker-pool dispatch
// mode" below. Config::worker_pool defaults to nullptr, in which case
// behavior is unchanged from Phase 3: handler_() runs synchronously inline
// on the reactor thread.
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
//     buffered in HttpConnection::base's bounded reactor::OutputQueue and
//     flushed on subsequent EPOLLOUT events
//     (reactor::ConnectionState::kWriting/kClosing) — see
//     docs/decisions/0006-buffer-representation.md (Phase 4) for the
//     write-queue formalization this replaced Phase 3's single flat buffer
//     with, and docs/decisions/0005-reactor-connection-model.md for the
//     read-pauses-while-writing policy underneath it.
//   - Not copyable or movable, matching BlockingHttpServer.
//
// Worker-pool dispatch mode (Phase 5 — see docs/decisions/0007-worker-
// pool-sizing.md for the full design rationale):
//   - connections_ holds std::shared_ptr<HttpConnection>, not unique_ptr,
//     specifically so a dispatched task can carry a std::weak_ptr instead
//     of a raw HttpConnection* or fd: a worker thread computes handler_()
//     purely against a *copy* of the parsed HttpRequest (never touching
//     HttpConnection/Connection, which are not thread-safe) and posts its
//     HttpResponse back through a thread-safe completion_queue_; the
//     reactor thread — and only the reactor thread — later calls
//     weak_ptr::lock() when draining that queue. If the connection was
//     already closed (client disconnected while its request was still
//     being handled on a worker thread), lock() safely returns nullptr and
//     the response is discarded — this is what avoids the fd-reuse hazard
//     a raw-fd-keyed lookup would have (a closed fd can be immediately
//     reassigned by accept() to an unrelated new connection; indexing
//     back into connections_ by the numeric fd alone could then deliver a
//     stale response to the wrong client).
//   - At most one request per connection is ever in flight on the worker
//     pool at a time (HttpConnection::awaiting_worker_result) — required
//     for correctness, not just simplicity: HTTP responses on a keep-alive/
//     pipelined connection must be written back in request order, but
//     worker threads complete in whatever order the pool schedules them.
//     drive_parser() stops parsing further pipelined requests on a
//     connection the instant it dispatches one to the pool, and only
//     resumes (from on_worker_result()) once that request's response has
//     been enqueued.
//   - If submit() to the pool is rejected (its bounded queue is full — FR5
//     "policies for ... a full worker queue"), the connection gets a 503
//     response with keep_alive forced false, exactly like any other
//     overload signal in this codebase (fail predictably, not silently).
//   - completion_queue_ (the channel a worker thread posts a finished
//     HttpResponse through) uses concurrency::BoundedWorkQueue::push(), the
//     *blocking* producer call, not try_push() — a momentarily-saturated
//     completion channel makes the posting worker thread wait for space
//     rather than silently dropping the response. This is safe specifically
//     because the producer here is a worker thread, never the reactor
//     thread (which still only ever uses non-blocking calls, e.g.
//     WorkerPool::submit()). Dropping here instead would leave
//     HttpConnection::awaiting_worker_result permanently true with nothing
//     left to ever clear it — a silent, permanent per-connection hang
//     reachable by ordinary configuration (completion_queue_capacity set
//     smaller than the worker pool's own concurrency), not just a wedged
//     pool. push() unblocks either once the reactor thread's next
//     on_completion_readable() drains some room, or once run() closes
//     completion_queue_ on the way out (see run()'s docs) — so a worker
//     is never stuck waiting past shutdown either.
//   - Caller-owned, not server-owned: `Config::worker_pool` is a
//     non-owning pointer. The caller must ensure the pointed-to
//     WorkerPool outlives this server, AND — critically — must call that
//     pool's stop() (joining every worker thread) before destroying this
//     server, if any submitted task might still be running. A worker
//     task's closure captures `this` (for handler_ and completion_queue_)
//     by reference; a worker thread still executing handler_() while this
//     server is destroyed is a use-after-free. Every test in
//     tests/integration/test_worker_dispatch_integration.cpp follows this
//     ordering (server stopped and joined first, then the pool) and is the
//     concrete reference for how to sequence it.
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
    // Per-connection buffer caps (spec decision D3 — see
    // docs/decisions/0006-buffer-representation.md). Defaults match
    // reactor::Connection's own defaults; overridable here so a caller
    // (e.g. a saturation test) can force the bound to bite deliberately.
    std::size_t max_output_queue_bytes = reactor::kDefaultMaxOutputQueueBytes;
    std::size_t max_read_buffer_bytes = reactor::kDefaultMaxReadBufferBytes;
    // Non-owning; nullptr (the default) means every request's handler_()
    // call runs inline on the reactor thread, exactly as in Phase 3. When
    // set, see "Worker-pool dispatch mode" in this class's docs — handler_
    // must be safe to call concurrently from multiple worker threads (no
    // shared mutable state without its own synchronization) since more
    // than one connection's request may be dispatched at once.
    concurrency::WorkerPool* worker_pool = nullptr;
    // Bound on how many completed-but-not-yet-delivered worker results may
    // accumulate before the reactor thread next drains them. Only
    // meaningful when worker_pool is set. Generous by default, and a real,
    // finite, configurable cap — never unbounded — matching every other
    // queue in this codebase; unlike those others, though, a *momentary*
    // saturation here does not reject or drop anything; it just makes the
    // posting worker thread wait (see "Worker-pool dispatch mode" above and
    // concurrency::BoundedWorkQueue::push()) until the reactor thread
    // catches up or the server starts shutting down. Set this low
    // deliberately only to test that backpressure path itself — an
    // unusually small value here mainly just slows worker threads down
    // under bursty completions, it does not cause dropped responses.
    std::size_t completion_queue_capacity = 4096;
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
  void on_listener_readable(int fd, std::uint32_t events);
  void on_client_event(int fd, std::uint32_t events);
  void on_completion_readable(int fd, std::uint32_t events);
  void process_readable(const std::shared_ptr<HttpConnection>& conn);
  void drive_parser(const std::shared_ptr<HttpConnection>& conn);
  void dispatch_to_worker(const std::shared_ptr<HttpConnection>& conn, protocol::HttpRequest request,
                           bool request_keep_alive);
  void on_worker_result(std::weak_ptr<HttpConnection> conn, protocol::HttpResponse response,
                         bool request_keep_alive);
  // Returns true iff the connection is still open and idle in kReading —
  // exactly the condition under which a caller may safely resume parsing
  // more pipelined data immediately (see on_worker_result()) rather than
  // waiting for the next reactor event.
  bool finish_dispatch(HttpConnection& conn, bool close_after_flush);
  void close_connection(int fd) noexcept;

  net::TcpListener listener_;
  reactor::EpollReactor reactor_;
  std::unordered_map<int, std::shared_ptr<HttpConnection>> connections_;
  RequestHandler handler_;
  Config config_;
  std::atomic<std::size_t> connections_accepted_{0};

  // Only constructed/registered when config_.worker_pool != nullptr — see
  // "Worker-pool dispatch mode" above.
  net::FileDescriptor completion_fd_;
  std::optional<concurrency::BoundedWorkQueue> completion_queue_;
};

}  // namespace arcserve::server
