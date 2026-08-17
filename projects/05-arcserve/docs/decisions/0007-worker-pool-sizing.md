# ADR 0007: Worker pool sizing (spec decision D5) and reactor-safe dispatch

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: 5 (worker pool)

## Context

Spec §1.9, decision D5: *"Worker affinity/count → default:
hardware_concurrency-derived, configurable; affinity is stretch."* FR4 asks
for "bounded request/work queues; configurable worker pool for CPU-bound
handler work (fixed worker count, bounded queue, safe stop semantics,
metrics); simple synchronous handler first." The Phase 5 roadmap row adds a
concrete deliverable beyond just building the pool: "direct-reactor vs
worker-dispatch comparison for a CPU-heavy handler."

D5's sizing question is the smaller half of this phase's design work. The
larger one, not named by any single spec decision but required to
implement D5/FR4 correctly at all, is: **how does a worker thread's
completed `HttpResponse` get back to the single-threaded reactor without
ever violating "a `Connection` is only touched from the reactor thread" or
introducing a fd-reuse hazard?** This ADR records both.

## Decision

### D5: sizing

`concurrency::default_worker_count()` (`include/arcserve/concurrency/worker_pool.hpp`)
returns `std::thread::hardware_concurrency()`, falling back to `1` in its
documented "not computable" case (`0`). `concurrency::WorkerPool::Config::worker_count`
defaults to `0`, meaning "use `default_worker_count()`" — configurable per
pool, exactly as D5 asks. Per-thread CPU affinity is out of scope: D5 itself
marks it a stretch goal, and nothing in this phase's exit criterion
("shutdown and saturation tests pass") needs it.

### FR4's other three requirements

- **Fixed worker count**: `WorkerPool`'s constructor spawns every worker
  thread immediately; there is no resizing after that. `worker_count()` is
  fixed for the pool's lifetime.
- **Bounded queue**: `concurrency::BoundedWorkQueue` (a thread-safe queue of
  `std::function<void()>` tasks, deliberately split out from `WorkerPool`
  itself so its push/pop/close/drain semantics are independently testable —
  see `tests/unit/test_bounded_work_queue.cpp`, including a many-producer/
  many-consumer stress test) caps `queue_capacity`; `submit()` never blocks
  the calling thread — a full queue returns `false` immediately, exactly
  the non-blocking-producer contract `reactor::OutputQueue::enqueue()`
  already established in ADR 0006, applied to work instead of bytes.
- **Safe stop semantics**: `WorkerPool::stop()` closes the queue (no more
  submissions accepted) and joins every worker thread, which each keep
  draining already-queued tasks to completion before observing the close —
  see `WorkerPoolTest.StopDrainsAlreadyQueuedTasksBeforeJoining`. No task is
  ever silently dropped by `stop()` itself, and no thread is ever detached
  or leaked. A task that throws is caught inside the worker loop (counted
  via `tasks_failed()`, never rethrown) — otherwise a single bad handler
  invocation would call `std::terminate` on a thread the caller never
  directly observes.
- **Metrics**: `tasks_completed()`, `tasks_failed()`, `queue_depth()` are
  the same kind of informal, test-facing counters `connections_accepted()`
  already is on both reactor-driven servers — Phase 7 formalizes real
  observability (FR8) for all of them together, not separately per
  component.

### Reactor-safe dispatch: the actual hard part

`NonblockingHttpServer::Config::worker_pool` (non-owning
`concurrency::WorkerPool*`, default `nullptr` — Phase 3 behavior, handler
runs inline) is how a caller opts a server into worker-dispatch mode. Four
coupled design choices make this safe:

1. **`shared_ptr`, not `unique_ptr`, in `connections_`.** A worker thread
   computes `handler_(request)` against a *copy* of the parsed
   `HttpRequest` — it never touches `HttpConnection`/`reactor::Connection`
   at all, both of which remain exactly as not-thread-safe as ADR 0005
   established. What it *does* need is a way to hand the resulting
   `HttpResponse` back to the connection it belongs to, safely, even if
   that connection closed in the meantime. A `std::weak_ptr<HttpConnection>`
   captured in the dispatched task's closure is exactly this: the worker
   thread only ever copies/moves the `weak_ptr` (itself a thread-safe
   operation on the shared control block), never calls `.lock()` on it.
   Only the reactor thread calls `.lock()`, when draining a completion —
   if the connection already closed, `.lock()` returns `nullptr` and the
   response is safely discarded.
2. **Why not key by raw fd instead?** A closed fd can be immediately
   reassigned by `accept()` to an unrelated new connection. Indexing back
   into `connections_` by the numeric fd alone, the way every other
   reactor-thread-only lookup in this codebase does, would risk delivering
   a stale worker's response to the *wrong* client if that race won. The
   `weak_ptr` approach makes this impossible by construction — it identifies
   the specific `HttpConnection` object, not a fd number that can be
   recycled.
3. **A thread-safe completion channel, woken via `eventfd`.** A worker
   thread finishing `handler_()` pushes a "deliver this response" closure
   onto `completion_queue_` (a `concurrency::BoundedWorkQueue` reused for
   this purpose — the same type, a different instance, with its own
   capacity) and writes to `completion_fd_`, an `eventfd` registered with
   the reactor exactly the way `EpollReactor`'s own internal `wakeup_fd_`
   wakes `run()` for `stop()`. `NonblockingHttpServer::on_completion_readable()`
   drains the eventfd counter and every currently-queued completion
   (`BoundedWorkQueue::drain()` — a non-blocking bulk-pop added
   specifically for this use, since the reactor thread must never block
   waiting on a queue) on the reactor thread. This is the only place a
   worker thread's result ever crosses back onto the thread that's allowed
   to touch `Connection` state.
4. **At most one worker-dispatched request per connection at a time.**
   `HttpConnection::awaiting_worker_result` is set the instant a request is
   dispatched and checked before `drive_parser()` is ever called again on
   that connection. This is a correctness requirement, not just simplicity:
   HTTP responses on a keep-alive/pipelined connection must be written back
   in request order, but a multi-worker pool completes tasks in whatever
   order it schedules them, not submission order. Gating to one in-flight
   request per connection is what keeps a fast second request from racing
   ahead of a slow first one on the wire.
   `WorkerDispatchIntegrationTest.PipelinedRequestsPreserveOrderDespiteVaryingHandlerDelay`
   is the regression test: it deliberately delays the *first* of two
   pipelined requests longer than the second and asserts the responses
   still arrive in send order.

### Overload: what happens when `submit()` fails

If the pool's bounded queue is already full, `submit()` returns `false`
immediately (never blocks the reactor thread). `dispatch_to_worker()`
responds with a synthesized `503 Service Unavailable` (`keep_alive` forced
`false`) via the exact same `on_worker_result()` path a real completion
uses — reusing that bookkeeping rather than duplicating it — so this
behaves identically to any other overload response from the connection's
perspective. `WorkerDispatchIntegrationTest.WorkerQueueSaturationRejectsExcessRequestsAndServerSurvives`
is the Phase 5 exit-criterion test: a pool with `worker_count = 1,
queue_capacity = 1`, a handler held blocked by the test, three concurrent
clients — the third gets a 503 while the first two eventually succeed once
released, and the server keeps serving new connections throughout.

### Ownership and shutdown ordering

`Config::worker_pool` is a non-owning pointer — a `WorkerPool` can outlive,
and be shared by, more than one server in principle, so `NonblockingHttpServer`
does not take ownership of it. This puts a real obligation on the caller,
documented prominently on the class: **stop the server (and join its
thread) before stopping the pool, and stop the pool before destroying the
server.** A submitted task's closure captures `this` (for `handler_` and
the completion channel) by reference; a worker thread still executing
`handler_()` while the server object is being destroyed would be a
use-after-free. Every fixture in
`tests/integration/test_worker_dispatch_integration.cpp`
(`WorkerDispatchFixture`) follows this exact ordering in its destructor and
is the concrete reference implementation for it.

### The direct-reactor-vs-worker-dispatch comparison

The roadmap's "direct-reactor vs worker-dispatch comparison for a
CPU-heavy handler" is satisfied as a correctness property, not a
benchmark: `WorkerDispatchIntegrationTest.CpuHeavyHandlerBlocksReactorWithoutWorkerPool`
and its sibling `...DoesNotBlockReactorWithWorkerPool` run the identical
150ms busy-wait handler (for one connection's request) and assert whether
a *different*, cheap request on a concurrently-connected client is delayed
by it — `> 100ms` without a pool configured, `< 80ms` with one. Per agent
execution rule 6 ("Never claim performance results without committed raw
benchmark data"), this is deliberately a bounded pass/fail assertion, not a
throughput/latency number — Phase 8's benchmark study is where this
comparison gets measured and reported with committed raw data.

## Consequences

- `NonblockingHttpServer::connections_` changed from
  `unordered_map<int, unique_ptr<HttpConnection>>` to
  `unordered_map<int, shared_ptr<HttpConnection>>` — a breaking internal
  change, scoped entirely to this class (nothing in `EchoReactorServer` or
  `reactor::Connection` needed it, since neither has a concept of
  dispatchable work).
- `HttpConnection` gains one field, `awaiting_worker_result`, always
  `false` when no worker pool is configured — Phase 3 behavior is
  unchanged when `Config::worker_pool` is left at its default `nullptr`,
  verified by every pre-existing Phase 3 integration test continuing to
  pass unmodified.
- A handler used with worker-dispatch mode must be safe to call
  concurrently from multiple worker threads (documented on
  `Config::worker_pool`) — a new, real constraint Phase 3's inline-only
  dispatch never had. `default_route` and this phase's test handlers all
  satisfy it (no shared mutable state without its own synchronization).
- Phase 6's idle-timeout sweep will need to account for a connection stuck
  in `awaiting_worker_result` (e.g. a worker pool wedged) — not yet handled,
  since Phase 6 owns timeouts.

## Amendment: two dispatch-correctness bugs found in review, fixed in place

A post-implementation review (Codex, high reasoning) found two BLOCKING
defects in the dispatch machinery this ADR describes, both fixed without
changing the decisions above:

1. **Redundant `finish_dispatch()` on the queue-full 503 fallback.** The
   "Overload: what happens when `submit()` fails" section above describes
   `dispatch_to_worker()`'s synchronous fallback as reusing
   `on_worker_result()` "so the ... bookkeeping isn't duplicated" — true for
   the bookkeeping, but `drive_parser()`'s caller-side guard
   (`if (conn.awaiting_worker_result) return;`) did not account for that
   fallback resolving synchronously: `on_worker_result()` already resets
   `awaiting_worker_result` to `false` and calls `finish_dispatch()` itself
   before `dispatch_to_worker()` returns, so `drive_parser()`'s guard failed
   to fire and it called `finish_dispatch()` a second, redundant time. If
   the connection's output still had bytes unflushed at the kernel level
   (e.g. a prior large response the peer hadn't drained yet, even though
   this connection's own `OutputQueue` had already reported itself fully
   flushed), that second call's own `flush_output()` could independently
   return `kPending` and stomp the connection's state from the correct
   `kClosing` back to `kWriting` — silently defeating a close the 503 had
   already promised the client on the wire. Fixed by tracking, in
   `drive_parser()` itself, whether the worker-dispatch branch was taken at
   all this call (`dispatched_to_worker_pool`) — covering both the
   genuinely-still-pending and the resolved-synchronously outcomes
   uniformly — rather than inferring it after the fact from
   `awaiting_worker_result`, which only distinguishes one of those two
   cases.
2. **A saturated `completion_queue_` used to strand the connection
   permanently.** `Config::completion_queue_capacity` is documented above as
   "bounded far more tightly already by worker_pool's own queue_capacity
   plus its worker_count" — that reasoning doesn't actually hold in
   general (workers keep completing and re-queuing new results without
   waiting for the reactor to drain old ones, so accumulated,
   not-yet-delivered completions aren't tightly bounded by the pool's own
   capacity), and nothing enforced any relationship between the two
   configs. A worker thread whose `completion_queue_->try_push()` failed
   used to silently drop the response and return — leaving
   `awaiting_worker_result` `true` forever, with nothing left to ever clear
   it: a silent, permanent per-connection hang reachable by ordinary
   configuration, not just a wedged pool. Fixed by giving
   `concurrency::BoundedWorkQueue` a second producer primitive, `push()`
   (blocking, unlike `try_push()`), and using it for this specific channel:
   safe specifically because the producer here is a worker thread, never
   the reactor thread (`WorkerPool::submit()` — the reactor thread's own
   producer call — is untouched and still non-blocking, exactly as FR5
   requires). `NonblockingHttpServer::run()` now closes `completion_queue_`
   right after the reactor loop exits, which wakes any worker still
   blocked in `push()` (returning `false`, response dropped) instead of
   letting it hang past shutdown and deadlock the caller's subsequent
   `WorkerPool::stop()` join.
