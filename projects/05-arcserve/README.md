# ArcServe

A C++20 event-driven network server, built up in phases: this repo
currently implements **Phase 0 (build/tooling foundation)** and **Phase 1
(blocking correctness server)** of the roadmap in the project spec
(`05-arcserve-spec.md` at the portfolio root). Later phases (nonblocking
epoll reactor, output buffering, worker pool, backpressure/timeouts,
observability, benchmarking) are not yet implemented — see
`docs/architecture.md`'s "What's deliberately not here yet" section.

## What's here

- **`arcserve_server`**: a single-threaded, blocking HTTP/1.1-subset
  server (`include/arcserve/server/blocking_server.hpp`). It handles one
  connection at a time, end to end, including keep-alive — its purpose is
  to validate the protocol/parser layer in isolation, before Phase 2
  introduces the nonblocking epoll reactor.
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
- `arcserve_unit_tests` — fd RAII, HTTP message serialization, and the
  parser state machine (including byte-at-a-time and every-split-offset
  fragmentation tests).
- `arcserve_integration_tests` — `BlockingHttpServer` driven over real
  loopback TCP sockets: fragmented requests, malformed input, oversized
  bodies, keep-alive, `Connection: close`, mid-request client disconnect,
  and sequential-client handling.

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
.github/workflows/ci.yml          Linux build+test (all sanitizer variants) + static analysis
```

## Decisions

Architectural choices and the spec's open decisions (D1, D6, decided ahead
of Phase 1 as the spec's handoff instructions require) are recorded in
[`docs/decisions/`](docs/decisions/).
