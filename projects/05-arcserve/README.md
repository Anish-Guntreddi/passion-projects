# ArcServe

A C++20 event-driven network server, built up in phases: this repo
currently implements **Phases 0–5** of the roadmap in the project spec
(`05-arcserve-spec.md` at the portfolio root) — build/tooling foundation,
a blocking correctness server, a nonblocking epoll reactor, an incremental
HTTP parser, bounded output buffering, and an optional worker pool. Later
phases (timeouts/backpressure policy, observability/profiling,
benchmarking) are not yet implemented — see `docs/architecture.md`'s
"What's deliberately not here yet" section.

## What's here

- **`arcserve_server`**: a single-threaded, blocking HTTP/1.1-subset
  server (`include/arcserve/server/blocking_server.hpp`). It handles one
  connection at a time, end to end, including keep-alive — its purpose is
  to validate the protocol/parser layer in isolation from the nonblocking
  epoll reactor below. (Still the production binary — see
  `docs/architecture.md`.)
- **`arcserve::server::NonblockingHttpServer`**: the same HTTP/1.1 subset
  served over a single-threaded `epoll` reactor
  (`include/arcserve/reactor/epoll_reactor.hpp`) instead of one blocking
  connection at a time — many connections make progress concurrently on
  one thread, none of them ever blocking it on socket I/O. Optionally
  dispatches request handling to a `concurrency::WorkerPool` instead of
  running inline — see "Worker pool" below.
- **`arcserve::reactor::Connection`** / **`arcserve::buffers::ByteBuffer`**
  / **`arcserve::reactor::OutputQueue`**: per-connection state with a
  contiguous, grow-capped read buffer and a bounded, multi-chunk output
  queue (spec decision D3 —
  [`docs/decisions/0006-buffer-representation.md`](docs/decisions/0006-buffer-representation.md)).
  A connection that overflows either bound is closed predictably; the
  server stays fully healthy for every other connection.
- **`arcserve::concurrency::WorkerPool`**: a fixed-size, bounded-queue
  thread pool for CPU-bound handler work
  (spec decision D5 —
  [`docs/decisions/0007-worker-pool-sizing.md`](docs/decisions/0007-worker-pool-sizing.md)).
  A saturated pool returns `503` predictably rather than blocking the
  reactor thread or growing memory without bound.
- **`arcserve::net::FileDescriptor`**: the RAII fd wrapper every socket in
  this codebase is owned by (`include/arcserve/net/file_descriptor.hpp`).
- **`arcserve::protocol::HttpRequestParser`**: a handwritten, byte-
  incremental HTTP/1.1-subset request parser
  (`include/arcserve/protocol/http_parser.hpp`). Scope:
  [`docs/protocol-scope.md`](docs/protocol-scope.md).
- Two routes for now: `GET /` (fixed text body) and `POST /echo` (echoes
  the request body back) — see `include/arcserve/server/default_handlers.hpp`.

## Build

Requires a C++20 compiler, CMake ≥ 3.20, and Ninja. Verified with g++ 13.3,
CMake 3.28.3, Ninja 1.11.1 on WSL2 Ubuntu 24.04. Network access is needed
once, at configure time, to fetch GoogleTest via `FetchContent`.

```bash
scripts/build.sh              # Debug build, no sanitizer
scripts/build.sh Debug asan   # Debug + AddressSanitizer + UndefinedBehaviorSanitizer
scripts/build.sh Debug tsan   # Debug + ThreadSanitizer
scripts/build.sh Release none # Release build
```

Binaries land in `build/<BuildType>-<sanitizer>/`.

## Test

```bash
scripts/test.sh               # builds (if needed) then runs the full suite via ctest
scripts/test.sh Debug asan    # same, under ASan+UBSan
```

Or directly: `ctest --test-dir build/Debug-none --output-on-failure`.

Two test binaries:
- `arcserve_unit_tests` — fd RAII, HTTP message serialization, the parser
  state machine (including byte-at-a-time and every-split-offset
  fragmentation tests), `EpollReactor`, `Connection`, `ByteBuffer`,
  `OutputQueue`, `BoundedWorkQueue`, `WorkerPool`.
- `arcserve_integration_tests` — `BlockingHttpServer`, `EchoReactorServer`,
  and `NonblockingHttpServer` (including its worker-pool dispatch mode)
  driven over real loopback TCP sockets: fragmented requests, malformed
  input, oversized bodies, keep-alive, `Connection: close`, mid-request
  client disconnect, slow-client partial writes, output-queue overflow,
  many-simultaneous-client stress, worker-pool saturation and shutdown,
  and a direct-reactor-vs-worker-dispatch comparison for a CPU-heavy
  handler.

## Run

```bash
build/Debug-none/arcserve_server [port]   # default port 8080
curl http://127.0.0.1:8080/
curl -d 'hello' http://127.0.0.1:8080/echo
```

`Ctrl-C` (SIGINT) or SIGTERM triggers a clean `stop()`.

## Repository layout

```
CMakeLists.txt, cmake/            build configuration (warnings, sanitizers)
include/arcserve/, src/           implementation (net, protocol, server, app)
tests/{unit,integration,support}/ test suites + the raw-socket test client
docs/                             architecture, protocol scope, ADRs
scripts/                          build.sh, test.sh, format.sh, lint.sh
.github/workflows/ci-arcserve.yml (repo root) Linux build+test (all sanitizer variants) + static analysis
```

## Decisions

Architectural choices and the spec's open decisions (D1–D6, each decided
ahead of the phase the spec's handoff instructions require: D1/D6 before
Phase 1, D2 before Phase 2, D3 before Phase 4, D5 during Phase 5) are
recorded in [`docs/decisions/`](docs/decisions/).
