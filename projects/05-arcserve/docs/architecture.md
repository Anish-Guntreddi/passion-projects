# Architecture (current state: through Phase 1)

This describes what exists after Phases 0–1. It will be extended (not
rewritten) as later phases land — see the roadmap in the project spec for
what each phase adds. Phase 9's "portfolio hardening" is where a full
request-lifecycle diagram is a deliverable artifact; this document is the
engineering reference in the meantime.

## Module layout

```
include/arcserve/
  net/file_descriptor.hpp     RAII fd wrapper — every fd's single owner (Phase 0)
  net/socket.hpp              blocking TcpListener, connect_loopback, read_some/write_all
  protocol/http_message.hpp   HttpRequest / HttpResponse / HeaderList / serialize()
  protocol/http_parser.hpp    HttpRequestParser — incremental HTTP/1.1-subset parser
  server/blocking_server.hpp  BlockingHttpServer — Phase 1's single-threaded server
  server/default_handlers.hpp default_route() — the GET // POST /echo reference handler

src/...                       matching .cpp implementations
src/app/main.cpp              process entry point (signal handling, port arg)

tests/unit/                   fd RAII, HTTP message serialization, parser state machine
tests/integration/            BlockingHttpServer over real sockets
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
`handle_connection` owns the accepted client `FileDescriptor` by value for
exactly the duration of that connection, on every return path (normal
completion, parse error, peer disconnect, or a write failure) — there is
no path that leaks or double-closes a socket. This is verified under
`-DARCSERVE_ENABLE_ASAN=ON` (leak checking is part of ASan on Linux) as
part of the test command in the project's phase report, not merely
asserted here.

## What's deliberately not here yet

- **epoll / nonblocking I/O** — Phase 2.
- **EPOLLOUT / partial-write-under-backpressure handling** beyond the
  simple `write_all()` retry loop — Phase 4 (slow-client behavior under a
  nonblocking reactor).
- **A worker pool / bounded work queue** — Phase 5. Phase 1's handlers run
  synchronously inline; that's fine because they're fast and fixed
  (`default_route`), and only one connection is ever mid-handler at a time.
- **Idle timeouts, overload backpressure, high-water marks** — Phase 6.
- **Metrics/structured logging, profiling** — Phase 7.
