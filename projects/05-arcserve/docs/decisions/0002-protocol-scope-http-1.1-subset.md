# ADR 0002: Protocol scope — HTTP/1.1 subset (spec decision D1)

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: Decided before Phase 1, per the spec's kickoff prompt ("D1 and
  D6 before Phase 1").

## Context

Spec §1.9, decision D1: *"HTTP subset vs custom framed protocol → default:
HTTP/1.1 subset (GET/POST, small header set, Content-Length bodies), ADR
the exact scope in `docs/protocol-scope.md`."* This is an open decision
with a recommended default; per the portfolio-wide rule to adopt each
spec's recommended default, this ADR adopts it as-is rather than
second-guessing it.

## Decision

Implement a narrow HTTP/1.1 request subset rather than a custom framed
protocol. The exact scope (methods, headers, limits, status codes) is
specified in [`docs/protocol-scope.md`](../protocol-scope.md) and enforced
by `HttpRequestParser`; this ADR records *why* HTTP over a custom protocol,
not the byte-level details.

### Why HTTP/1.1 over a custom framed protocol

- **Load-generation tooling exists off the shelf.** The tech stack plan
  (§Part 2) calls for `wrk`/`wrk2` for the Phase 8 benchmark study.
  Choosing HTTP means Phase 8 doesn't also need a custom load generator to
  be written and validated — that would be extra unmeasured code between
  the server and the benchmark numbers, working against "never fabricate
  benchmark... numbers" by adding a second thing that could be silently
  wrong.
- **The deliverable artifacts (§1.5) are more legible as HTTP.** A
  throughput-vs-concurrency graph and a request-lifecycle diagram read
  clearly to any reviewer when the unit of work is "an HTTP request";
  they'd need extra explanation for a bespoke frame format.
- **The resume narrative (§1.5)** is written around "nonblocking Linux
  sockets, epoll, bounded worker queues, explicit backpressure" —
  properties that are protocol-agnostic. HTTP doesn't cost anything
  relative to a custom protocol in terms of what's actually being
  demonstrated (the event loop, the backpressure policy, the worker pool);
  it only adds request-line/header/Content-Length parsing, which is itself
  explicitly in-scope engineering work (FR3).
- **A custom framed protocol would need its own spec written from
  scratch** with no external reference to check the implementation
  against; RFC 9110/9112 give a concrete, narrow-able source of truth for
  what "correct" means for the subset chosen.

### Why a narrow *subset*, not full HTTP/1.1

Per the spec's non-goals (§1.4: "Full RFC-complete HTTP") and Part 4 rule
2 ("do not jump to... until the epoll baseline is correct and measured" —
the same discipline applies to protocol surface: don't build parsing
generality the project doesn't need yet). Concretely cut from full
HTTP/1.1: chunked transfer-encoding, methods beyond GET/POST, absolute-
/authority-/asterisk-form targets, trailers, `Expect: 100-continue`, range
requests. Each of these is a meaningful chunk of parser state-machine
complexity for a feature no test or benchmark in this project's roadmap
requires. `docs/protocol-scope.md` is the enforcement boundary: anything
not listed there is rejected, predictably, rather than partially handled.

## Consequences

- `wrk`/`wrk2` (or any standard HTTP client, including `curl`, for manual
  debugging) can drive ArcServe directly — no bespoke client is needed
  anywhere in this project, including in the test suite (the integration
  tests use a minimal raw-socket test client only to control byte-level
  framing/fragmentation on the *request* side, deliberately, not because a
  real HTTP client wouldn't otherwise work).
- Every "unsupported" input has a defined, tested HTTP status code
  response (400/411/413/431/501/505) rather than an ambiguous close or
  hang — this is directly testable and is exercised by both the unit
  parser tests and the integration tests.
- If a future phase needs chunked encoding or another method, that is a
  new ADR extending `docs/protocol-scope.md`, not an undocumented parser
  change.
