# Architecture (current state: through Phase 3)

This describes what exists after Phases 0–3. It will be extended (not
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
  protocol/http_message.hpp      HttpRequest / HttpResponse / HeaderList / serialize()
  protocol/http_parser.hpp       HttpRequestParser — incremental HTTP/1.1-subset parser
  reactor/epoll_reactor.hpp      EpollReactor — protocol-agnostic level-triggered epoll
                                  wrapper + graceful-shutdown wakeup (Phase 2)
  reactor/connection.hpp         Connection — generic per-connection state (FR2): owned
                                  fd, read/write buffers, lifecycle state, peer, timestamp
  server/request_handler.hpp     RequestHandler — shared by every HTTP server variant
  server/blocking_server.hpp     BlockingHttpServer — Phase 1's single-threaded server
  server/default_handlers.hpp    default_route() — the GET // POST /echo reference handler
  server/echo_reactor_server.hpp EchoReactorServer — Phase 2's protocol-agnostic reactor
                                  demonstrator (raw byte echo, no HTTP)
  server/http_connection.hpp     HttpConnection — reactor::Connection + HttpRequestParser
                                  (Phase 3)
  server/nonblocking_http_server.hpp
                                  NonblockingHttpServer — Phase 3's reactor-driven HTTP server

src/...                       matching .cpp implementations (reactor::Connection and
                               server::HttpConnection are header-only — trivial enough
                               not to need one)
src/app/main.cpp              process entry point (signal handling, port arg) — still
                               wired to BlockingHttpServer; switching the production
                               binary to NonblockingHttpServer is a later-phase decision,
                               not made yet

tests/unit/                   fd RAII, nonblocking socket primitives, HTTP message
                               serialization, parser state machine, EpollReactor,
                               reactor::Connection
tests/integration/            BlockingHttpServer, EchoReactorServer, and
                               NonblockingHttpServer, all over real sockets
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
                          -> try_flush(): write_all() once
                               fully flushed & not closing -> stay/return to kReading
                               fully flushed & closing     -> close_connection()
                               partial (WouldBlock)        -> kWriting/kClosing, arm EPOLLOUT
  client writable   -> try_flush() again (resumes draining write_buffer)
  EPOLLERR/EPOLLHUP -> close_connection() unconditionally
```

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
write failure). In Phases 2–3, the same guarantee holds through a
different shape: each accepted client fd is owned by exactly one
`reactor::Connection` (Phase 2) or `server::HttpConnection` (Phase 3), in
turn owned by exactly one entry in that server's `connections_` registry
(`std::unordered_map<int, std::unique_ptr<...>>`); `close_connection()` is
the single path that erases a registry entry (and, via the owned
`FileDescriptor`'s destructor, closes the fd) and is called from every
place a connection ends. There is no path in either server that leaks or
double-closes a socket. This is verified under `-DARCSERVE_ENABLE_ASAN=ON`
(leak checking is part of ASan on Linux) as part of the test command in
the project's phase report, not merely asserted here.

## What's deliberately not here yet

- **A real multi-message write queue with configurable high-water marks**
  — Phase 4. Phases 2–3's `Connection::write_buffer` is a single buffer
  that already handles partial writes and EPOLLOUT correctly (see
  `docs/decisions/0005-reactor-connection-model.md`), but it isn't yet the
  formalized, observable backpressure policy FR5/Phase 6 describe.
- **A worker pool / bounded work queue** — Phase 5. Handlers
  (`RequestHandler`) still run synchronously inline on whichever thread is
  driving the connection — the single serving thread in
  `BlockingHttpServer`, the single reactor thread in
  `NonblockingHttpServer` — in both cases fine so far because
  `default_route` is fast and fixed.
- **Idle timeouts, overload backpressure, high-water marks** — Phase 6.
  `reactor::Connection::last_activity()`/`touch()` exist for this already
  (FR2), but nothing reads them yet.
- **Metrics/structured logging, profiling** — Phase 7.
  `connections_accepted()` on both reactor-driven servers is an informal,
  test-facing counter, not FR8's real observability story.
- **D3's buffer representation ADR** — owed before Phase 4, not yet
  written; `read_buffer`/`write_buffer` are plain `std::string`s in the
  meantime (see ADR 0005, decision 4).
