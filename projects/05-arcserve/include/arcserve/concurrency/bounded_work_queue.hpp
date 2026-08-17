#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace arcserve::concurrency {

// A bounded, thread-safe queue of tasks — the "bounded ... work queue"
// FR4 asks for, split out from WorkerPool because it is independently
// interesting and independently testable: the bounded-push/blocking-pop/
// safe-shutdown-wakeup behavior is exactly the kind of concurrency-heavy
// code spec section 1.6 asks to have "TSan-compatible tests", and testing
// it directly (pushing/popping tasks with no thread actually running them)
// keeps those tests fast and deterministic rather than timing-sensitive.
//
// Concurrency: every public method is safe to call concurrently from any
// number of threads, any combination, at any time.
class BoundedWorkQueue {
 public:
  using Task = std::function<void()>;

  explicit BoundedWorkQueue(std::size_t capacity) : capacity_(capacity) {}

  BoundedWorkQueue(const BoundedWorkQueue&) = delete;
  BoundedWorkQueue& operator=(const BoundedWorkQueue&) = delete;
  BoundedWorkQueue(BoundedWorkQueue&&) = delete;
  BoundedWorkQueue& operator=(BoundedWorkQueue&&) = delete;
  ~BoundedWorkQueue() = default;

  // Attempts to push `task` onto the back of the queue. Returns false
  // immediately (never blocks, and never moves-from `task`) if the queue is
  // already at capacity() or close() has already been called — a bounded
  // queue with a non-blocking producer side (FR4/FR5: "an unbounded queue
  // is a review-blocking defect", and a producer that blocked indefinitely
  // on a full work queue would just turn the backpressure problem into a
  // hang instead of an observable rejection the caller can act on). This is
  // the right contract for a producer that must never block — e.g. the
  // reactor thread calling WorkerPool::submit() — because *that* thread has
  // nowhere else to apply backpressure except "reject predictably now".
  [[nodiscard]] bool try_push(Task task);

  // Blocks the calling thread until either `task` is enqueued (returns
  // true) or close() is called before space becomes available (returns
  // false without enqueuing `task`). Unlike try_push(), this is for a
  // producer where waiting *is* the desired behavior — concretely, a
  // WorkerPool worker thread delivering a completed result onto a
  // downstream bounded channel (see NonblockingHttpServer::completion_queue_)
  // where dropping the result on a momentary saturation would otherwise
  // strand whatever was waiting on it forever, and where the thread that
  // *would* have blocked is never the reactor thread submit()'s contract
  // exists to protect. Safe to call concurrently with any other method,
  // from any thread; never leaves `task` partially moved-from on the
  // false/closed path.
  [[nodiscard]] bool push(Task task);

  // Blocks the calling thread until either a task is available (returns
  // it, FIFO order) or close() has been called and the queue is empty
  // (returns std::nullopt) — the signal a worker thread uses to know it's
  // time to exit.
  [[nodiscard]] std::optional<Task> pop();

  // Removes and returns every task currently queued, without blocking —
  // used by a non-worker consumer (Phase 5's cross-thread completion
  // channel in NonblockingHttpServer) that must never block the reactor
  // thread waiting for work to arrive, only drain whatever is already
  // there once notified out-of-band (e.g. by an eventfd write). Returns an
  // empty vector if nothing was queued; safe to call on a closed queue.
  [[nodiscard]] std::vector<Task> drain();

  // Marks the queue closed: every blocked/future pop() on an empty queue
  // returns std::nullopt instead of blocking forever, and every future
  // try_push() fails. Does not discard tasks already queued — pop()/
  // drain() still return them. Idempotent; safe to call from any thread,
  // any number of times.
  void close() noexcept;

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool closed() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable not_empty_or_closed_;
  // Signaled whenever a slot frees up (pop()/drain() removes something) or
  // close() is called — what push() waits on. Kept separate from
  // not_empty_or_closed_ (a different predicate) rather than reused, so a
  // push() waiter is never spuriously woken by a pop()-side notification or
  // vice versa.
  std::condition_variable not_full_or_closed_;
  std::deque<Task> tasks_;
  std::size_t capacity_;
  bool closed_ = false;
};

}  // namespace arcserve::concurrency
