# Architecture (current state: through Phase 5)

This describes what exists after Phases 0–5. It will be extended (not
rewritten) as later phases land — see the roadmap in the project spec for
what each phase adds. Phase 9's "portfolio hardening" is where a full
request-lifecycle diagram is a deliverable artifact; this document is the
engineering reference in the meantime.

## Module layout

```
include/arcserve/
  net/file_descriptor.hpp        RAII fd wrapper — every fd's single owner (Phase 0)
  net/socket.hpp                 TcpListener (blocking + nonblocking), connect_loopback,
                                  set_nonblocking, describe_peer, read_some/write_all
  buffers/byte_buffer.hpp        ByteBuffer — contiguous, grow-capped byte buffer
                                  (spec decision D3, Phase 4; docs/decisions/0006)
  protocol/http_message.hpp      HttpRequest / HttpResponse / HeaderList / serialize()
  protocol/http_parser.hpp       HttpRequestParser — incremental HTTP/1.1-subset parser
  reactor/epoll_reactor.hpp      EpollReactor — protocol-agnostic level-triggered epoll
                                  wrapper + graceful-shutdown wakeup (Phase 2)
  reactor/output_queue.hpp       OutputQueue — bounded, multi-chunk send queue built on
                                  ByteBuffer (Phase 4; docs/decisions/0006)
  reactor/connection.hpp         Connection — generic per-connection state (FR2): owned
                                  fd, ByteBuffer read_buffer, OutputQueue output, lifecycle
                                  state, peer, timestamp, flush_output()
  concurrency/bounded_work_queue.hpp
                                  BoundedWorkQueue — thread-safe bounded task queue
                                  (Phase 5; docs/decisions/0007)
  concurrency/worker_pool.hpp    WorkerPool — fixed-size worker-thread pool over a
                                  BoundedWorkQueue (Phase 5; docs/decisions/0007)
  server/request_handler.hpp     RequestHandler — shared by every HTTP server variant
  server/blocking_server.hpp     BlockingHttpServer — Phase 1's single-threaded server
  server/default_handlers.hpp    default_route() — the GET // POST /echo reference handler
  server/echo_reactor_server.hpp EchoReactorServer — Phase 2's protocol-agnostic reactor
                                  demonstrator (raw byte echo, no HTTP)
  server/http_connection.hpp     HttpConnection — reactor::Connection + HttpRequestParser
                                  + awaiting_worker_result (Phase 3, extended Phase 5)
  server/nonblocking_http_server.hpp
                                  NonblockingHttpServer — Phase 3's reactor-driven HTTP
                                  server, with Phase 5's optional worker-pool dispatch mode

src/...                       matching .cpp implementations (server::HttpConnection stays
                               header-only — trivial enough not to need one; reactor::
                               Connection gained one, src/reactor/connection.cpp, for
                               flush_output())
src/app/main.cpp              process entry point (signal handling, port arg) — still
                               wired to BlockingHttpServer; switching the production
                               binary to NonblockingHttpServer (with or without a worker
                               pool) is a later-phase decision, not made yet

tests/unit/                   fd RAII, nonblocking socket primitives, HTTP message
                               serialization, parser state machine, EpollReactor,
                               reactor::Connection, ByteBuffer, OutputQueue,
                               BoundedWorkQueue, WorkerPool
tests/integration/            BlockingHttpServer, EchoReactorServer, NonblockingHttpServer
                               (including its worker-pool dispatch mode), all over real
                               sockets
tests/support/                RawTcpClient — a raw-socket test client (test-only)
```

## Request flow (Phase 1: blocking, single connection at a time)

```
accept() (via a bounded poll() loop, so stop() is observed promptly)
  -> handle_connection(client fd)
       loop:
         parser.feed(leftover-and-newly-read bytes)
           kNeedMoreData -> read_some() off the socket, append, loop
           kError        -> write the parser's canned error response, close
           kComplete     -> handler(request) -> HttpResponse
                             write_all() the serialized response
                             keep-alive? loop (parser.reset()) : close
```

There is **no concurrency inside the server yet**: `BlockingHttpServer`
handles exactly one connection fully (including its whole keep-alive
sequence) before accepting the next. New connections queue in the kernel's
`listen()` backlog while that happens. This is deliberate scope, not an
oversight — see the roadmap: the nonblocking epoll reactor (Phase 2) is
what removes this limitation, and Phase 1 exists specifically to validate
protocol/parser correctness *before* that complexity is introduced, so a
bug found later can be isolated to the reactor rather than the parser.

## Request flow (Phases 2–3: nonblocking epoll reactor)

```
EpollReactor::run() -> epoll_wait(-1) (unbounded; woken by stop()'s eventfd)
  listener readable -> accept_nonblocking() in a loop until nullopt
                          -> new HttpConnection, reactor_.add(client_fd, kReadable, ...)
  client readable   -> process_readable(): read_some() in a loop until WouldBlock/short read
                          -> drive_parser(): parser.feed(view) in a loop
                               kNeedMoreData -> stash leftover in read_buffer, wait for next event
                               kError        -> queue canned error response, mark close-after-flush
                               kComplete     -> handler(request) -> HttpResponse, queue it,
                                                 parser.reset(), loop (may resolve a pipelined
                                                 next request already in the same buffer)
                          -> flush_output(): drains the OutputQueue chunk by chunk
                               fully flushed & not closing -> stay/return to kReading
                               fully flushed & closing     -> close_connection()
                               partial (WouldBlock)        -> kWriting/kClosing, arm EPOLLOUT
  client writable   -> flush_output() again (resumes draining the output queue)
  EPOLLERR/EPOLLHUP -> close_connection() unconditionally
```

(Phase 4 renamed the per-server `try_flush()` — near-identically duplicated
in `EchoReactorServer` and `NonblockingHttpServer` through Phase 3 — into
one shared `Connection::flush_output()`; see
`docs/decisions/0006-buffer-representation.md`.)

## Request flow (Phase 5: optional worker-pool dispatch)

When `NonblockingHttpServer::Config::worker_pool` is set, `drive_parser()`'s
`kComplete` branch dispatches instead of calling `handler_()` inline:

```
kComplete -> move the parsed HttpRequest out, reset the parser,
             mark awaiting_worker_result = true, submit a task to the pool
             (never blocks the reactor thread), return — this connection's
             parsing pauses until its response comes back.

  [worker thread] handler_(request) -> HttpResponse
                   -> push a completion closure onto completion_queue_
                   -> write 1 to completion_fd_ (an eventfd registered
                      with the reactor, exactly like EpollReactor's own
                      internal stop() wakeup)

  [reactor thread] completion_fd_ readable -> drain completion_queue_
                   -> for each completion: weak_ptr::lock() the connection
                        expired -> discard (connection closed meanwhile)
                        alive   -> enqueue_output(), flush_output(),
                                   awaiting_worker_result = false,
                                   resume drive_parser() if still kReading
                                   (any pipelined bytes queued while
                                   waiting get parsed now, in order)
```

If `submit()` is rejected (the pool's bounded queue is full), the
connection gets a `503` through this same completion path instead —
overload is a predictable response, not a hang or unbounded growth. See
`docs/decisions/0007-worker-pool-sizing.md` for the full design, including
why `connections_` holds `shared_ptr<HttpConnection>` (so a
`weak_ptr`-based hand-off is possible at all) and why at most one request
per connection is ever in flight on the pool (response ordering on
pipelined/keep-alive connections).

Unlike Phase 1, many connections are in flight at once, each parked in
whichever of `{reading, writing, closing}` its own state machine
(`reactor::ConnectionState`) says it's in — nothing here ever blocks
waiting on one connection's socket I/O while another connection has work
to do (Phase 2's `EchoReactorServer` and Phase 3's `NonblockingHttpServer`
are both built this way; see `docs/decisions/0005-reactor-connection-model.md`
for the read-pauses-while-writing policy this depends on, and
`docs/decisions/0004-epoll-level-triggered.md` for why the reactor is
level- rather than edge-triggered). `EchoReactorServer` runs the identical
`EpollReactor`/`Connection` machinery with no `protocol::` dependency at
all — concrete evidence that the reactor layer doesn't secretly know
anything about HTTP.

`NonblockingHttpServer` reuses `HttpRequestParser` and `RequestHandler`
completely unchanged from Phase 1: the same parser, the same
`default_route()` handler, the same `docs/protocol-scope.md` contract, the
same status-code behavior — only how bytes get in and out of the parser
(one blocking connection at a time vs. many connections multiplexed on one
reactor thread) differs between `BlockingHttpServer` and
`NonblockingHttpServer`. This is directly verified: every correctness test
in `tests/integration/test_blocking_server_integration.cpp` has a matching
test in `tests/integration/test_nonblocking_http_server_integration.cpp`
asserting the identical observable behavior.

## Why the parser takes `std::string_view&`

`HttpRequestParser::feed(std::string_view& data)` mutates `data` in place
(via `remove_prefix`) to reflect exactly how many bytes were consumed.
This is what makes two things fall out for free, without special-casing:

- **Fragmented input** (FR3): calling `feed()` with a 1-byte view, in a
  loop, is exactly as correct as calling it once with the whole request —
  the parser's internal state (`line_buffer_`, `header_bytes_seen_`,
  `content_length_`) carries everything needed across calls. This is
  tested directly (`HandlesByteAtATimeFragmentation`,
  `HandlesSplitAtEveryOffset` in `tests/unit/test_http_parser.cpp`) and
  end-to-end over real sockets (`FragmentedRequestArrivesCorrectly` in the
  integration suite).
- **Pipelining**: if a single `recv()` returns bytes for more than one
  request (e.g. two small keep-alive requests sent back-to-back), `feed()`
  stops as soon as the first request completes and leaves the rest in
  `data` for the caller. `BlockingHttpServer::handle_connection` treats
  that leftover exactly like unread socket bytes — it's fed to the (reset)
  parser before any new `read_some()` call.

## Ownership and fd lifetime

Every file descriptor in this codebase is owned by exactly one
`arcserve::net::FileDescriptor` at a time (Phase 0's RAII wrapper):
`TcpListener` owns the listening socket for the server's entire lifetime;
`BlockingHttpServer::handle_connection` owns the accepted client
`FileDescriptor` by value for exactly the duration of that connection, on
every return path (normal completion, parse error, peer disconnect, or a
write failure). In Phases 2–5, the same guarantee holds through a
different shape: each accepted client fd is owned by exactly one
`reactor::Connection` (Phase 2) or `server::HttpConnection` (Phase 3), in
turn owned by exactly one entry in that server's `connections_` registry —
`std::unordered_map<int, std::unique_ptr<Connection>>` in
`EchoReactorServer`, `std::unordered_map<int, std::shared_ptr<HttpConnection>>`
in `NonblockingHttpServer` (`shared_ptr` since Phase 5, specifically so a
worker-dispatched request can carry a `std::weak_ptr` back to its
connection without risking a stale-fd delivery — see
`docs/decisions/0007-worker-pool-sizing.md`). `close_connection()` is the
single path that erases a registry entry (and, via the owned
`FileDescriptor`'s destructor, closes the fd once the last reference to it
drops) and is called from every place a connection ends. There is no path
in either server that leaks or double-closes a socket. This is verified
under `-DARCSERVE_ENABLE_ASAN=ON` (leak checking is part of ASan on Linux)
as part of the test command in the project's phase report, not merely
asserted here.

## What's deliberately not here yet

- **Configurable, observable backpressure *policy*** — Phase 6. Phase 4
  (`docs/decisions/0006-buffer-representation.md`) gave every buffer a real,
  enforced cap and one fixed response to exceeding it (close the
  connection); Phase 6 is where that becomes soft-vs-hard marks, a
  proactive response instead of a silent drop-then-close where feasible,
  and rejection counters (FR8).
- **Idle timeouts, graceful overload shutdown** — Phase 6.
  `reactor::Connection::last_activity()`/`touch()` exist for this already
  (FR2, and are still updated on every read/write, including through
  `flush_output()`), but nothing sweeps them yet. A connection stuck in
  `awaiting_worker_result` forever (e.g. a wedged worker pool) is a
  concrete case Phase 6's sweep needs to cover.
- **Metrics/structured logging, profiling** — Phase 7.
  `connections_accepted()`, `WorkerPool::tasks_completed()`/
  `tasks_failed()`/`queue_depth()` are informal, test-facing counters, not
  FR8's real observability story.
- **A production binary that actually uses the worker pool** —
  `src/app/main.cpp` still wires up `BlockingHttpServer` with no worker
  pool at all; switching it to `NonblockingHttpServer` (optionally with a
  `WorkerPool`) is a later-phase decision, not made in Phase 5.
- **Per-thread CPU affinity for worker threads** — spec decision D5 itself
  marks this a stretch goal (`docs/decisions/0007-worker-pool-sizing.md`).
- **io_uring, custom allocators, TLS** — explicitly out of scope until the
  epoll baseline is measured (agent execution rule 2) / MVP non-goals
  (spec §1.4).
