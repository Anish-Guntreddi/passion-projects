#include "arcserve/concurrency/bounded_work_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace arcserve::concurrency {
namespace {

TEST(BoundedWorkQueueTest, TryPushSucceedsUntilCapacity) {
  BoundedWorkQueue queue(2);
  EXPECT_TRUE(queue.try_push([] {}));
  EXPECT_TRUE(queue.try_push([] {}));
  EXPECT_FALSE(queue.try_push([] {}));  // at capacity
  EXPECT_EQ(queue.size(), 2u);
  EXPECT_EQ(queue.capacity(), 2u);
}

TEST(BoundedWorkQueueTest, PopReturnsTasksInFifoOrder) {
  BoundedWorkQueue queue(4);
  std::vector<int> order;
  ASSERT_TRUE(queue.try_push([&order] { order.push_back(1); }));
  ASSERT_TRUE(queue.try_push([&order] { order.push_back(2); }));
  ASSERT_TRUE(queue.try_push([&order] { order.push_back(3); }));

  for (int i = 0; i < 3; ++i) {
    std::optional<BoundedWorkQueue::Task> task = queue.pop();
    ASSERT_TRUE(task.has_value());
    (*task)();
  }
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(BoundedWorkQueueTest, PopBlocksUntilATaskIsPushed) {
  BoundedWorkQueue queue(4);
  std::atomic<bool> ran{false};

  std::thread popper([&queue, &ran] {
    std::optional<BoundedWorkQueue::Task> task = queue.pop();
    ASSERT_TRUE(task.has_value());
    (*task)();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let popper block on an empty queue
  EXPECT_FALSE(ran.load());
  ASSERT_TRUE(queue.try_push([&ran] { ran.store(true); }));

  popper.join();
  EXPECT_TRUE(ran.load());
}

TEST(BoundedWorkQueueTest, CloseWakesABlockedPopWithNullopt) {
  BoundedWorkQueue queue(4);
  std::optional<bool> popped_has_value;

  std::thread popper([&queue, &popped_has_value] {
    std::optional<BoundedWorkQueue::Task> task = queue.pop();
    popped_has_value = task.has_value();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  queue.close();
  popper.join();

  ASSERT_TRUE(popped_has_value.has_value());
  EXPECT_FALSE(*popped_has_value);
  EXPECT_TRUE(queue.closed());
}

TEST(BoundedWorkQueueTest, CloseStillDeliversAlreadyQueuedTasksBeforeReturningNullopt) {
  BoundedWorkQueue queue(4);
  ASSERT_TRUE(queue.try_push([] {}));
  ASSERT_TRUE(queue.try_push([] {}));
  queue.close();

  EXPECT_TRUE(queue.pop().has_value());
  EXPECT_TRUE(queue.pop().has_value());
  EXPECT_FALSE(queue.pop().has_value());  // drained and closed
}

TEST(BoundedWorkQueueTest, TryPushFailsAfterClose) {
  BoundedWorkQueue queue(4);
  queue.close();
  EXPECT_FALSE(queue.try_push([] {}));
}

TEST(BoundedWorkQueueTest, PushSucceedsImmediatelyWhenSpaceIsAvailable) {
  BoundedWorkQueue queue(2);
  EXPECT_TRUE(queue.push([] {}));
  EXPECT_TRUE(queue.push([] {}));
  EXPECT_EQ(queue.size(), 2u);
}

// The behavior that fixes the completion-channel stranding bug: push()
// waits for a pop() elsewhere to free a slot instead of failing outright —
// unlike try_push(), which would return false immediately here.
TEST(BoundedWorkQueueTest, PushBlocksUntilSpaceFreesUpThenSucceeds) {
  BoundedWorkQueue queue(1);
  ASSERT_TRUE(queue.try_push([] {}));  // fill the only slot
  std::atomic<bool> pushed{false};

  std::thread pusher([&queue, &pushed] {
    EXPECT_TRUE(queue.push([] {}));
    pushed.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let pusher block on a full queue
  EXPECT_FALSE(pushed.load());

  ASSERT_TRUE(queue.pop().has_value());  // frees the slot the first try_push() took
  pusher.join();
  EXPECT_TRUE(pushed.load());
  EXPECT_EQ(queue.size(), 1u);
}

// The shutdown-safety half of the fix: a worker thread blocked in push()
// must be woken by close() (returning false) rather than hanging forever —
// see NonblockingHttpServer::run()'s completion_queue_->close() call, which
// exists specifically so this can never deadlock a caller's subsequent
// WorkerPool::stop() join.
TEST(BoundedWorkQueueTest, CloseWakesABlockedPushWithFalse) {
  BoundedWorkQueue queue(1);
  ASSERT_TRUE(queue.try_push([] {}));  // fill the only slot; never popped
  std::optional<bool> push_result;

  std::thread pusher([&queue, &push_result] { push_result = queue.push([] {}); });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  queue.close();
  pusher.join();

  ASSERT_TRUE(push_result.has_value());
  EXPECT_FALSE(*push_result);
  EXPECT_EQ(queue.size(), 1u);  // the blocked task was never enqueued
}

TEST(BoundedWorkQueueTest, PushFailsImmediatelyAfterClose) {
  BoundedWorkQueue queue(4);
  queue.close();
  EXPECT_FALSE(queue.push([] {}));
}

TEST(BoundedWorkQueueTest, DrainReturnsEverythingQueuedWithoutBlocking) {
  BoundedWorkQueue queue(4);
  std::vector<int> order;
  ASSERT_TRUE(queue.try_push([&order] { order.push_back(1); }));
  ASSERT_TRUE(queue.try_push([&order] { order.push_back(2); }));

  std::vector<BoundedWorkQueue::Task> drained = queue.drain();
  EXPECT_EQ(drained.size(), 2u);
  EXPECT_EQ(queue.size(), 0u);
}

// Concurrency stress: many producers racing try_push against the queue's
// bounded capacity, many consumers racing pop(), then close() — every task
// that was successfully pushed must run exactly once, and no thread may
// hang. This is the kind of test spec section 1.6 calls out ("Concurrency:
// worker shutdown, queue saturation, TSan-compatible tests").
TEST(BoundedWorkQueueTest, ManyProducersAndConsumersProcessEveryPushedTaskExactlyOnce) {
  BoundedWorkQueue queue(16);
  constexpr int kProducers = 8;
  constexpr int kPerProducer = 200;
  std::atomic<int> pushed{0};
  std::atomic<int> processed{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&queue, &pushed] {
      for (int i = 0; i < kPerProducer; ++i) {
        while (!queue.try_push([&pushed] { pushed.fetch_add(1, std::memory_order_relaxed); })) {
          std::this_thread::yield();  // bounded queue full; retry (backpressure)
        }
      }
    });
  }

  constexpr int kConsumers = 4;
  std::vector<std::thread> consumers;
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&queue, &processed] {
      for (;;) {
        std::optional<BoundedWorkQueue::Task> task = queue.pop();
        if (!task.has_value()) {
          return;
        }
        (*task)();
        processed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  queue.close();
  for (auto& t : consumers) {
    t.join();
  }

  EXPECT_EQ(pushed.load(), kProducers * kPerProducer);
  EXPECT_EQ(processed.load(), kProducers * kPerProducer);
}

}  // namespace
}  // namespace arcserve::concurrency
