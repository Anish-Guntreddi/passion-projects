# ADR 0005: Reactor connection model and Phase 2–3 buffering scope

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: 2–3 (nonblocking epoll reactor; incremental HTTP parser
  integration)

## Context

Phase 2's deliverable is "Reactor + connection registry; echo under many
connections" (exit: "No blocking I/O in reactor; stress test passes").
Phase 3's is "HTTP subset... partial-read tests, malformed input, limits"
(exit: "Fuzz/property cases where practical; end-to-end requests succeed").
Neither phase's exit criterion requires the fully-formalized
output-buffering/backpressure machinery the roadmap assigns to later
phases (Phase 4: "EPOLLOUT management, write queues, correct partial-write
and slow-client behavior"; Phase 5: worker pool; Phase 6: "high-water marks
and policies"). This ADR records the concrete scope line drawn for this
implementation and why later phases can build on top of it rather than
needing to replace it.

## Decisions

1. **`reactor::EpollReactor` is entirely protocol-agnostic.** It knows
   about fds, an interest mask (`kReadable`/`kWritable`), and callbacks —
   nothing about HTTP, buffers, or connections. `reactor::Connection` (the
   generic per-connection object: owned fd, `read_buffer`/`write_buffer`,
   `ConnectionState`, peer metadata, `last_activity` timestamp) is likewise
   protocol-agnostic. `EchoReactorServer` (Phase 2) uses `reactor::Connection`
   directly; `server::HttpConnection` (Phase 3) *composes* a
   `reactor::Connection` with `protocol::HttpRequestParser` rather than the
   reactor layer knowing HTTP exists. This is deliberate, concrete evidence
   for the spec's acceptance criterion "architecture explainable without
   Boost.Asio hiding the event model" (§1.8) — the reactor genuinely
   doesn't hide or presuppose any particular protocol; `EchoReactorServer`
   is proof, not just an assertion, since it never includes anything from
   `arcserve/protocol/`.

2. **One write buffer per connection, not a queue of discrete messages.**
   `Connection::write_buffer` is a single growing/shrinking `std::string`;
   a response (or, for pipelined HTTP requests resolved in one reactor
   event, several responses back to back) is appended to it, and
   `try_flush()` drains from its front on every writable event. This is
   sufficient for both Phase 2's echo demonstrator and Phase 3's
   `NonblockingHttpServer` — neither ever needs to distinguish "response N"
   from "response N+1" once serialized, they're just bytes on the wire in
   order — and is what actually gets tested end-to-end
   (`LargeTransferSurvivesPartialWrites`,
   `LargeTransferDoesNotBlockOtherClients` in
   `tests/integration/test_echo_reactor_server_integration.cpp`). Phase 4's
   roadmap item is explicitly to formalize this into a real multi-message
   write queue; this ADR is what that phase's implementer should read
   first to understand what's already correct here (partial-write
   draining, EPOLLOUT re-arming) versus what's new work there (configurable
   high-water marks, per-message backpressure signaling to the
   request-producing side).

3. **Reading pauses while a write is draining.** Once a connection has any
   bytes queued in `write_buffer` that didn't fit in one `send(2)`, its
   epoll interest becomes `kWritable` only
   (`ConnectionState::kWriting`/`kClosing`) until the buffer fully drains,
   at which point interest returns to `kReadable` (`kReading`). This is a
   deliberate, minimal backpressure behavior — *not* Phase 6's formal
   "high-water marks and policies" (FR5), which will need to be
   configurable and observable (queue-depth metrics, rejection counters).
   It exists now because letting `write_buffer` grow without bound while
   still accepting more input from the same slow peer would violate the
   spec's core design principle ("the server fails gracefully under
   overload instead of growing memory without bound") even before Phase 6
   gets around to policy-izing it. It also means a peer that both sends a
   large amount of data *and* refuses to read its own responses can stall
   its own connection (bounded by the OS's receive-buffer/TCP-window
   behavior on that one connection) — this is expected and is exactly why
   `LargeTransferSurvivesPartialWrites` and
   `LargeTransferDoesNotBlockOtherClients` both use a concurrent
   reader/writer thread pair on the client side rather than a strictly
   sequential send-then-receive script, which would otherwise deadlock
   against this very policy instead of testing it.

4. **Buffer representation stays `std::string` (D3 remains open).** Spec
   decision D3 ("buffer representation... ADR before Phase 4") is
   explicitly not due yet. `read_buffer`/`write_buffer` are ordinary
   growable `std::string`s — correct and simple, not the contiguous
   grow-capped byte-buffer design D3's default describes, and not
   optimized for the allocation/copy behavior Phase 7's profiling will
   care about. This is a known, intentional placeholder, not an oversight:
   D3's ADR is still owed before Phase 4 begins.

## Consequences

- Phases 4–6 extend `reactor::Connection`/`server::HttpConnection` and the
  servers built on them; none of Phase 2–3's public APIs (`EpollReactor`,
  `Connection`, `HttpConnection`, `NonblockingHttpServer`) need a breaking
  rewrite to add a real write queue, high-water marks, or a worker pool —
  those are additive.
- Nothing in Phases 2–3 measures or claims a performance number (per rule
  6) — the pause-reads-while-writing policy above is a
  correctness/boundedness argument, not a throughput one; Phase 8 is where
  any of this gets benchmarked.
- A future contributor implementing Phase 4 should not be surprised that
  `write_buffer` is "just a string" — that's this ADR's decisions 2 and 4,
  not a bug to fix quietly.
