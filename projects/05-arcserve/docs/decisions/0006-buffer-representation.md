# ADR 0006: Buffer representation (spec decision D3) and the Phase 4 write queue

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: Decided before Phase 4, per the spec's kickoff prompt ("D3
  before Phase 4") and ADR 0005 decision 4, which left it explicitly open.

## Context

Spec §1.9, decision D3: *"Buffer representation → default: contiguous
grow-capped byte buffers; ADR before Phase 4."* ADR 0005 (Phase 2–3)
recorded that `reactor::Connection::read_buffer`/`write_buffer` were
`std::string`s in the meantime — correct, but neither contiguous-by-design
nor capped — and that Phase 4's roadmap item ("EPOLLOUT management, write
queues, correct partial-write and slow-client behavior") was where this ADR
and the real write-queue formalization both belonged.

Two related but separable questions needed answers:

1. **D3 itself**: what does a single buffer look like?
2. **What Phase 4 called "write queues"**: `write_buffer` was one flat
   buffer that every serialized response got concatenated onto — correct
   (partial writes and EPOLLOUT re-arming already worked, per ADR 0005),
   but neither bounded nor genuinely multi-message.

## Decision

### D3: `buffers::ByteBuffer`

`include/arcserve/buffers/byte_buffer.hpp` / `src/buffers/byte_buffer.cpp`
implement the spec's recommended default directly: a contiguous,
grow-capped byte buffer backed by one `std::vector<char>` plus a read
offset.

- **Contiguous**: `data()`/`size()` always span one unbroken block of
  unconsumed bytes, so every read/write syscall site in this codebase
  (`net::read_some`/`net::write_all`) still hands a single pointer+length
  to the kernel — no scatter/gather machinery was introduced.
- **Grow-capped**: `append()` is all-or-nothing against a caller-supplied
  `max_capacity()` — it either fully succeeds or leaves the buffer
  completely unchanged and returns `false`. Never partial, never silently
  unbounded.
- **Compaction, not a ring buffer**: `consume()` advances a read offset and
  reclaims the consumed prefix's storage once keeping it around stops being
  worth an extra branch (`compact_if_worthwhile()` in the `.cpp`: an O(1)
  `clear()` on the common "fully drained" case, an O(remaining) shift once
  more than half the backing storage is consumed but not all of it). A true
  ring buffer was considered and rejected: this codebase's dominant access
  pattern (`ByteBufferTest.ManyPartialConsumeCyclesAtSteadyStateNeverFail`
  is the regression test for this) is "accumulate, then drain to (at or
  near) empty," which the simpler append-at-back/consume-at-front shape
  already handles cheaply — a ring buffer's wraparound bookkeeping would
  buy nothing here and would cost the contiguity property (a wrapped ring
  buffer's live region generally isn't one contiguous span).

`reactor::Connection::read_buffer` is now a `buffers::ByteBuffer`, capped at
`kDefaultMaxReadBufferBytes` (8 MiB) by default and configurable per
connection/server. This is genuinely a new bound, not just a type change:
Phase 2–3's `std::string read_buffer` could grow without limit if a peer's
kernel receive buffer ever handed the reactor an unusually large single
burst; now it can't, and `EchoReactorServer`/`NonblockingHttpServer` both
close the connection predictably (never crash, never grow past the cap) if
`append()` ever returns `false` — see
`EchoReactorServerIntegrationTest.OutputQueueOverflowClosesConnectionAndServerSurvives`'s
sibling read-side handling in `process_readable()`.

### The Phase 4 write queue: `reactor::OutputQueue`

`include/arcserve/reactor/output_queue.hpp` / `src/reactor/output_queue.cpp`
replace `Connection::write_buffer` (a flat `std::string`) with a bounded,
multi-chunk queue of `buffers::ByteBuffer`s:

- `enqueue(data)` adds one discrete chunk (typically one serialized
  `HttpResponse`, or one echoed read for `EchoReactorServer`) behind
  whatever is already queued, capped at an aggregate `max_buffered_bytes()`
  — `kDefaultMaxOutputQueueBytes` (16 MiB) by default, configurable per
  connection/server, same all-or-nothing bounded contract as `ByteBuffer`.
- `front()`/`consume()` let a caller drain across chunk boundaries one
  `send(2)` loop at a time without ever needing to merge chunks into one
  contiguous span — the wire doesn't care about chunk boundaries, only
  byte order, so `Connection::flush_output()` (new: this logic used to be
  duplicated as `try_flush()` in both `EchoReactorServer` and
  `NonblockingHttpServer`; it is now one method on `Connection` itself)
  just calls `net::write_all()` against the front chunk, `consume()`s
  whatever it sent, and moves to the next chunk once the front one is
  fully drained.

This is deliberately **not** FR5/Phase 6's full backpressure *policy*
machinery (soft vs. hard high-water marks, pause-vs-reject, observable
rejection counters). It enforces exactly one hard cap and reports
`true`/`false`; every caller in this codebase today does the same thing
with a `false` return: close the connection once whatever is already queued
drains. That is intentionally the whole policy for now — see "Consequences"
below for what's still Phase 6's to add.

### Why bounding the write queue is new work, not cosmetic

Before this ADR, `write_buffer` could grow without any configured limit:
a synchronous handler producing many pipelined responses (or, after Phase
5, several worker-dispatched ones) against a client that stopped reading
its own socket could grow server memory for exactly as long as that one
connection stayed open. That is precisely the case agent execution rule 4
("Bounded queues everywhere — an unbounded queue is a review-blocking
defect") and the spec's core design principle ("the server fails
gracefully under overload instead of growing memory without bound") rule
out. `OutputQueue` closes that gap in the minimal, correct way available at
this phase: a real, enforced, configurable cap plus a predictable failure
mode (connection close), covered end-to-end by
`EchoReactorServerIntegrationTest.OutputQueueOverflowClosesConnectionAndServerSurvives`
and
`NonblockingHttpServerIntegrationTest.OutputQueueOverflowClosesConnectionAndServerSurvives`
— both assert the overflowing connection closes *and* that the server
stays fully healthy for every other connection afterward.

## Consequences

- `reactor::Connection`'s public surface changes: `write_buffer` (a public
  `std::string` field) is gone, replaced by `output_queue()` (mutable/const
  accessors onto the `OutputQueue`), `enqueue_output()` (a convenience
  wrapper), and `flush_output()` (the now-shared drain logic). `read_buffer`
  keeps its name and public-field visibility but changes type from
  `std::string` to `buffers::ByteBuffer`. Every caller in this codebase
  (`EchoReactorServer`, `NonblockingHttpServer`, `tests/unit/test_connection.cpp`)
  was updated accordingly; this is a breaking API change scoped entirely to
  Phase 4, exactly as ADR 0005 predicted.
- `EchoReactorServer::try_flush()` and `NonblockingHttpServer::try_flush()`
  (near-identical, hand-duplicated code) are both deleted in favor of the
  one `Connection::flush_output()` — a direct simplification this phase's
  work made possible, not merely incidental to it.
- Phase 6 ("Timeouts/backpressure/shutdown... high-water marks and
  policies... overload test shows bounded memory/queue behavior") still has
  real work: today's policy is a single hard cap with one fixed response
  (close). Phase 6 is where soft marks (e.g. proactively stop *reading* a
  slow connection's requests once its output queue crosses a soft
  threshold, rather than only ever rejecting the response that finally
  crosses the hard one), a proactive 503 instead of a silent drop-then-close
  where feasible, and observable rejection counters (FR8) belong.
- Phase 5's worker pool (`concurrency::WorkerPool`) builds directly on
  `enqueue_output()`'s bounded, `bool`-returning contract for its own
  "submit failed" case — see docs/decisions/0007-worker-pool-sizing.md.
