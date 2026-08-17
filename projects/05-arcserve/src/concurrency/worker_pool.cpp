#include "arcserve/concurrency/worker_pool.hpp"

#include <exception>
#include <utility>

namespace arcserve::concurrency {

std::size_t default_worker_count() noexcept {
  unsigned int detected = std::thread::hardware_concurrency();
  return detected == 0 ? 1 : static_cast<std::size_t>(detected);
}

namespace {
std::size_t resolve_worker_count(std::size_t configured) {
  return configured == 0 ? default_worker_count() : configured;
}
}  // namespace

WorkerPool::WorkerPool(Config config) : queue_(config.queue_capacity) {
  std::size_t count = resolve_worker_count(config.worker_count);
  workers_.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

WorkerPool::~WorkerPool() { stop(); }

bool WorkerPool::submit(BoundedWorkQueue::Task task) { return queue_.try_push(std::move(task)); }

void WorkerPool::stop() noexcept {
  if (stopped_.exchange(true, std::memory_order_acq_rel)) {
    return;  // Already stopped (or stopping) — idempotent.
  }
  queue_.close();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void WorkerPool::worker_loop() {
  for (;;) {
    std::optional<BoundedWorkQueue::Task> task = queue_.pop();
    if (!task.has_value()) {
      return;  // Queue closed and drained — this worker's job is done.
    }
    try {
      (*task)();
      tasks_completed_.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      // Swallowed deliberately — see class docs: an exception escaping a
      // detached worker thread would call std::terminate. Counted so it is
      // at least observable rather than silently lost.
      tasks_failed_.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

}  // namespace arcserve::concurrency
