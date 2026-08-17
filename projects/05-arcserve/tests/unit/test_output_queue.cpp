#include "arcserve/reactor/output_queue.hpp"

#include <cstddef>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

namespace arcserve::reactor {
namespace {

TEST(OutputQueueTest, DefaultConstructedIsEmpty) {
  OutputQueue queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.queued_bytes(), 0u);
  EXPECT_EQ(queue.queued_chunks(), 0u);
  EXPECT_EQ(queue.front(), "");
}

TEST(OutputQueueTest, EnqueueAddsAChunk) {
  OutputQueue queue;
  EXPECT_TRUE(queue.enqueue("hello"));
  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.queued_chunks(), 1u);
  EXPECT_EQ(queue.queued_bytes(), 5u);
  EXPECT_EQ(queue.front(), "hello");
}

TEST(OutputQueueTest, MultipleChunksStayDistinctUntilConsumed) {
  OutputQueue queue;
  ASSERT_TRUE(queue.enqueue("abc"));
  ASSERT_TRUE(queue.enqueue("defg"));
  EXPECT_EQ(queue.queued_chunks(), 2u);
  EXPECT_EQ(queue.queued_bytes(), 7u);
  EXPECT_EQ(queue.front(), "abc");  // front() never spans a chunk boundary

  queue.consume(3);  // fully drains the front chunk
  EXPECT_EQ(queue.queued_chunks(), 1u);
  EXPECT_EQ(queue.queued_bytes(), 4u);
  EXPECT_EQ(queue.front(), "defg");

  queue.consume(4);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.queued_bytes(), 0u);
}

TEST(OutputQueueTest, PartialConsumeLeavesRemainderOfFrontChunk) {
  OutputQueue queue;
  ASSERT_TRUE(queue.enqueue("abcdef"));
  queue.consume(2);
  EXPECT_EQ(queue.front(), "cdef");
  EXPECT_EQ(queue.queued_bytes(), 4u);
  EXPECT_EQ(queue.queued_chunks(), 1u);  // still one chunk, just partially drained
}

TEST(OutputQueueTest, ConsumeClampedToAvailableBytesAndIsANoOpWhenEmpty) {
  OutputQueue queue;
  queue.consume(50);  // nothing queued; must not crash or underflow
  EXPECT_TRUE(queue.empty());

  ASSERT_TRUE(queue.enqueue("ab"));
  queue.consume(100);  // clamped
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.queued_bytes(), 0u);
}

TEST(OutputQueueTest, EnqueueRejectedWhenOverCapacityLeavesQueueUnchanged) {
  OutputQueue queue(/*max_buffered_bytes=*/10);
  ASSERT_TRUE(queue.enqueue("12345"));   // 5 bytes
  ASSERT_TRUE(queue.enqueue("67890"));   // exactly at the 10-byte cap
  EXPECT_FALSE(queue.enqueue("x"));      // would push to 11

  EXPECT_EQ(queue.queued_bytes(), 10u);
  EXPECT_EQ(queue.queued_chunks(), 2u);
  EXPECT_EQ(queue.max_buffered_bytes(), 10u);
}

TEST(OutputQueueTest, CapacityFreedByConsumeAllowsFurtherEnqueues) {
  OutputQueue queue(/*max_buffered_bytes=*/5);
  ASSERT_TRUE(queue.enqueue("abcde"));
  EXPECT_FALSE(queue.enqueue("f"));  // full

  queue.consume(5);
  EXPECT_TRUE(queue.enqueue("xyz"));
  EXPECT_EQ(queue.front(), "xyz");
}

TEST(OutputQueueTest, EnqueuingEmptyDataIsANoOpSuccess) {
  OutputQueue queue(/*max_buffered_bytes=*/0);
  EXPECT_TRUE(queue.enqueue(std::string_view()));
  EXPECT_TRUE(queue.empty());
}

// Draining across a chunk boundary within a single logical byte stream:
// consuming exactly the front chunk's remaining size pops it and the very
// next front() call reflects the next chunk with no gap or duplication.
TEST(OutputQueueTest, ThreeChunksDrainInOrderByteForByte) {
  OutputQueue queue;
  ASSERT_TRUE(queue.enqueue("A"));
  ASSERT_TRUE(queue.enqueue("BB"));
  ASSERT_TRUE(queue.enqueue("CCC"));

  std::string_view drained;
  std::string collected;
  while (!queue.empty()) {
    drained = queue.front();
    collected.append(drained);
    queue.consume(drained.size());
  }
  EXPECT_EQ(collected, "ABBCCC");
}

}  // namespace
}  // namespace arcserve::reactor
