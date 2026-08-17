#include "arcserve/concurrency/worker_pool.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

namespace arcserve::concurrency {
namespace {

TEST(WorkerPoolTest, DefaultWorkerCountIsPositiveAndMatchesHardwareConcurrencyWhenKnown) {
  std::size_t count = default_worker_count();
  EXPECT_GT(count, 0u);
  unsigned int hw = std::thread::hardware_concurrency();
  if (hw != 0) {
    EXPECT_EQ(count, static_cast<std::size_t>(hw));
  }
}

TEST(WorkerPoolTest, ConfiguredWorkerCountIsRespected) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 3, .queue_capacity = 16});
  EXPECT_EQ(pool.worker_count(), 3u);
  pool.stop();
}

TEST(WorkerPoolTest, ZeroWorkerCountFallsBackToDefault) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 0, .queue_capacity = 16});
  EXPECT_EQ(pool.worker_count(), default_worker_count());
  pool.stop();
}

// A small helper: a thread-safe "wait until N tasks have run" latch, used
// instead of a fixed sleep so this test is not timing-dependent/flaky.
class CountLatch {
 public:
  explicit CountLatch(int target) : target_(target) {}
  void count_down() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++count_;
    if (count_ >= target_) {
      cv_.notify_all();
    }
  }
  bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return count_ >= target_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_ = 0;
  int target_;
};

TEST(WorkerPoolTest, SubmittedTasksRunAndAreCounted) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 4, .queue_capacity = 64});
  constexpr int kTasks = 50;
  CountLatch latch(kTasks);

  for (int i = 0; i < kTasks; ++i) {
    ASSERT_TRUE(pool.submit([&latch] { latch.count_down(); }));
  }

  ASSERT_TRUE(latch.wait_for(std::chrono::seconds(5)));
  pool.stop();
  EXPECT_EQ(pool.tasks_completed(), static_cast<std::size_t>(kTasks));
  EXPECT_EQ(pool.tasks_failed(), 0u);
}

TEST(WorkerPoolTest, ExceptionInATaskIsCountedAndDoesNotCrashThePool) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 2, .queue_capacity = 16});
  CountLatch latch(1);

  ASSERT_TRUE(pool.submit([] { throw std::runtime_error("boom"); }));
  ASSERT_TRUE(pool.submit([&latch] { latch.count_down(); }));

  ASSERT_TRUE(latch.wait_for(std::chrono::seconds(5)));
  pool.stop();
  EXPECT_EQ(pool.tasks_failed(), 1u);
  EXPECT_EQ(pool.tasks_completed(), 1u);
}

// "Safe stop semantics" (FR4): stop() must let already-queued tasks run to
// completion, not abandon them — this is what distinguishes a graceful
// drain from simply killing worker threads.
TEST(WorkerPoolTest, StopDrainsAlreadyQueuedTasksBeforeJoining) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 1, .queue_capacity = 32});
  constexpr int kTasks = 20;
  std::atomic<int> completed{0};

  for (int i = 0; i < kTasks; ++i) {
    ASSERT_TRUE(pool.submit([&completed] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      completed.fetch_add(1, std::memory_order_relaxed);
    }));
  }

  pool.stop();  // must not return until all 20 have run
  EXPECT_EQ(completed.load(), kTasks);
}

TEST(WorkerPoolTest, StopIsIdempotent) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 2, .queue_capacity = 16});
  pool.stop();
  pool.stop();  // must not hang, double-join, or crash
  SUCCEED();
}

TEST(WorkerPoolTest, SubmitAfterStopFails) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 1, .queue_capacity = 16});
  pool.stop();
  EXPECT_FALSE(pool.submit([] {}));
}

TEST(WorkerPoolTest, DestructorStopsRunningThreads) {
  std::atomic<int> completed{0};
  {
    WorkerPool pool(WorkerPool::Config{.worker_count = 2, .queue_capacity = 16});
    for (int i = 0; i < 5; ++i) {
      ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1, std::memory_order_relaxed); }));
    }
  }  // destructor must join every worker before returning
  EXPECT_EQ(completed.load(), 5);
}

// Bounded queue: with exactly one worker kept busy on a task the test
// controls the release of, and a queue capacity of exactly one, a third
// submission must be rejected — proving the "full worker queue" bound is
// real, not just configured and ignored.
TEST(WorkerPoolTest, SubmitFailsWhenQueueIsFull) {
  WorkerPool pool(WorkerPool::Config{.worker_count = 1, .queue_capacity = 1});

  std::mutex mutex;
  std::condition_variable release_cv;
  bool released = false;
  std::atomic<bool> first_task_running{false};

  ASSERT_TRUE(pool.submit([&] {
    first_task_running.store(true);
    std::unique_lock<std::mutex> lock(mutex);
    release_cv.wait(lock, [&released] { return released; });
  }));

  // Wait for the first task to actually be picked up by the sole worker
  // (not just sitting in the queue), so the queue itself is genuinely
  // empty when the second submit() below fills it.
  for (int i = 0; i < 1000 && !first_task_running.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(first_task_running.load());

  EXPECT_TRUE(pool.submit([] {}));   // fills the 1-capacity queue
  EXPECT_FALSE(pool.submit([] {}));  // queue full and the worker is still busy

  {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
  }
  release_cv.notify_all();
  pool.stop();
}

}  // namespace
}  // namespace arcserve::concurrency
