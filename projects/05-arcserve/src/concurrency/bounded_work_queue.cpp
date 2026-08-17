#include "arcserve/concurrency/bounded_work_queue.hpp"

#include <utility>

namespace arcserve::concurrency {

bool BoundedWorkQueue::try_push(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || tasks_.size() >= capacity_) {
      return false;
    }
    tasks_.push_back(std::move(task));
  }
  not_empty_or_closed_.notify_one();
  return true;
}

bool BoundedWorkQueue::push(Task task) {
  std::unique_lock<std::mutex> lock(mutex_);
  not_full_or_closed_.wait(lock, [this] { return tasks_.size() < capacity_ || closed_; });
  if (closed_) {
    return false;  // Never enqueued; nothing will ever pop()/drain() this.
  }
  tasks_.push_back(std::move(task));
  lock.unlock();
  not_empty_or_closed_.notify_one();
  return true;
}

std::optional<BoundedWorkQueue::Task> BoundedWorkQueue::pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_or_closed_.wait(lock, [this] { return !tasks_.empty() || closed_; });
  if (tasks_.empty()) {
    return std::nullopt;  // closed_ and drained
  }
  Task task = std::move(tasks_.front());
  tasks_.pop_front();
  lock.unlock();
  not_full_or_closed_.notify_one();  // A slot just freed up for push() to claim.
  return task;
}

std::vector<BoundedWorkQueue::Task> BoundedWorkQueue::drain() {
  std::vector<Task> drained;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    drained.reserve(tasks_.size());
    for (auto& task : tasks_) {
      drained.push_back(std::move(task));
    }
    tasks_.clear();
  }
  if (!drained.empty()) {
    not_full_or_closed_.notify_all();  // Freed however many slots were drained.
  }
  return drained;
}

void BoundedWorkQueue::close() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  not_empty_or_closed_.notify_all();
  not_full_or_closed_.notify_all();
}

std::size_t BoundedWorkQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tasks_.size();
}

bool BoundedWorkQueue::closed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_;
}

}  // namespace arcserve::concurrency
