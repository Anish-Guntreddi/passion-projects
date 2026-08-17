#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "arcserve/concurrency/bounded_work_queue.hpp"

namespace arcserve::concurrency {

// std::thread::hardware_concurrency()'s value, or 1 if the platform can't
// report it (its documented "0 if not computable/well defined" case) —
// spec decision D5's recommended default; see
// docs/decisions/0007-worker-pool-sizing.md.
[[nodiscard]] std::size_t default_worker_count() noexcept;

// A fixed-size pool of worker threads draining a BoundedWorkQueue (FR4:
// "configurable worker pool for CPU-bound handler work — fixed worker
// count, bounded queue, safe stop semantics, metrics"). Construction spawns
// every worker thread immediately; there is no dynamic resizing — a fixed
// count is exactly what FR4 and D5 ask for (D5 marks per-thread CPU
// affinity a stretch goal, out of scope here).
//
// Concurrency/lifetime invariants:
//   - submit() is safe to call concurrently from any thread (in this
//     codebase, always the single reactor thread of whichever server owns
//     this pool — see NonblockingHttpServer's worker-dispatch mode — but
//     nothing here assumes that).
//   - Every worker thread runs the identical loop: pop() a task from the
//     queue (blocking) and invoke it, until pop() returns std::nullopt
//     (queue closed and drained) — see BoundedWorkQueue's docs.
//   - stop() closes the queue and joins every worker thread, so it returns
//     only once every already-queued-and-not-yet-started task has run to
//     completion and every worker thread has exited — "safe stop
//     semantics" (FR4): no task is silently dropped, and no worker thread
//     is ever detached or leaked. Idempotent (a second call is a no-op).
//     The destructor calls stop() unconditionally, so a WorkerPool going
//     out of scope never leaves threads running. Must not be called from
//     one of this pool's own worker threads (would deadlock joining
//     itself — the same restriction std::thread::join() itself has).
//   - A task that throws is caught by the worker loop (counted via
//     tasks_failed(), never rethrown/propagated) — an uncaught exception
//     escaping a detached-from-the-caller worker thread would otherwise
//     invoke std::terminate and take the whole process down over one bad
//     handler.
//   - Any caller submitting a task that captures state shared with code
//     running on another thread (e.g. NonblockingHttpServer's `this` and
//     its `handler_`) must ensure that state outlives every worker thread
//     that might still be running such a task — concretely, stop() this
//     pool (or otherwise guarantee no submitted task is still executing)
//     before destroying anything a submitted task's closure captured by
//     reference or raw pointer. See docs/decisions/0007-worker-pool-
//     sizing.md for how NonblockingHttpServer's worker-dispatch mode
//     satisfies this without that ordering constraint ever touching
//     per-connection state directly.
//   - Not copyable or movable: identity (its threads, its queue) must not
//     be duplicated or relocated while running.
class WorkerPool {
 public:
  struct Config {
    std::size_t worker_count = 0;  // 0 = default_worker_count()
    std::size_t queue_capacity = 1024;
  };

  // No default argument here deliberately: a nested Config's own default
  // member initializers are not usable as a default *argument* value at
  // this point in the class body (a GCC/Clang-agnostic, standard-mandated
  // ordering — verified against this toolchain; see docs/decisions/0007-
  // worker-pool-sizing.md). Every constructor call in this codebase passes
  // Config explicitly (e.g. `WorkerPool pool(WorkerPool::Config{});`),
  // matching the same convention BlockingHttpServer/EchoReactorServer/
  // NonblockingHttpServer already use for their own nested Config.
  explicit WorkerPool(Config config);

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;
  WorkerPool(WorkerPool&&) = delete;
  WorkerPool& operator=(WorkerPool&&) = delete;
  ~WorkerPool();

  // Attempts to enqueue `task` for some worker to run. Returns false
  // (without enqueuing anything) if the bounded queue is already full or
  // stop() has already been called — see BoundedWorkQueue::try_push().
  // Never blocks the calling thread, by design: this is exactly what keeps
  // a full work queue from turning into "the calling thread now also
  // blocks on ordinary work", which the spec's core design principle rules
  // out for the reactor thread specifically.
  [[nodiscard]] bool submit(BoundedWorkQueue::Task task);

  // Stops accepting new work, lets every already-queued task run to
  // completion, joins every worker thread, then returns. See class docs
  // for the full invariant.
  void stop() noexcept;

  [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }

  // Informal counters for tests/Phase 7's future formal observability
  // story (FR8/FR4's "metrics"); see class docs.
  [[nodiscard]] std::size_t tasks_completed() const noexcept {
    return tasks_completed_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t tasks_failed() const noexcept {
    return tasks_failed_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t queue_depth() const { return queue_.size(); }

 private:
  void worker_loop();

  BoundedWorkQueue queue_;
  std::vector<std::thread> workers_;
  std::atomic<std::size_t> tasks_completed_{0};
  std::atomic<std::size_t> tasks_failed_{0};
  std::atomic<bool> stopped_{false};
};

}  // namespace arcserve::concurrency
