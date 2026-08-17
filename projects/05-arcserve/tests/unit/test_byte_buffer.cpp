#include "arcserve/buffers/byte_buffer.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

namespace arcserve::buffers {
namespace {

TEST(ByteBufferTest, DefaultConstructedIsEmpty) {
  ByteBuffer buf;
  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.max_capacity(), kUnboundedCapacity);
}

TEST(ByteBufferTest, AppendGrowsSizeAndIsRetrievableContiguously) {
  ByteBuffer buf;
  ASSERT_TRUE(buf.append(std::string_view("hello")));
  ASSERT_TRUE(buf.append(std::string_view(" world")));

  EXPECT_EQ(buf.size(), 11u);
  EXPECT_EQ(buf.view(), "hello world");
  // Contiguity: data() + size() spans the whole logical content in one
  // unbroken block, so a caller can pass it directly to a syscall.
  EXPECT_EQ(std::string(buf.data(), buf.size()), "hello world");
}

TEST(ByteBufferTest, ConsumeRemovesFromFront) {
  ByteBuffer buf;
  ASSERT_TRUE(buf.append(std::string_view("abcdef")));

  buf.consume(2);
  EXPECT_EQ(buf.view(), "cdef");

  buf.consume(100);  // clamped to size(), not UB
  EXPECT_TRUE(buf.empty());
}

TEST(ByteBufferTest, AppendAfterFullConsumeStartsFresh) {
  ByteBuffer buf;
  ASSERT_TRUE(buf.append(std::string_view("first")));
  buf.consume(5);
  ASSERT_TRUE(buf.empty());

  ASSERT_TRUE(buf.append(std::string_view("second")));
  EXPECT_EQ(buf.view(), "second");
}

TEST(ByteBufferTest, ClearDropsEverything) {
  ByteBuffer buf;
  ASSERT_TRUE(buf.append(std::string_view("data")));
  buf.clear();
  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(ByteBufferTest, AppendRejectedWhenOverCapacityLeavesBufferUnchanged) {
  ByteBuffer buf(/*max_capacity=*/5);
  ASSERT_TRUE(buf.append(std::string_view("abc")));

  // "abc" (3) + "de" (2) == 5, exactly at the cap: allowed.
  EXPECT_TRUE(buf.append(std::string_view("de")));
  EXPECT_EQ(buf.view(), "abcde");

  // One more byte would exceed it: rejected, buffer left exactly as-is
  // (all-or-nothing — never a partial append).
  EXPECT_FALSE(buf.append(std::string_view("f")));
  EXPECT_EQ(buf.view(), "abcde");
  EXPECT_EQ(buf.size(), 5u);
}

TEST(ByteBufferTest, ZeroCapacityRejectsAnyNonEmptyAppend) {
  ByteBuffer buf(/*max_capacity=*/0);
  EXPECT_TRUE(buf.append(std::string_view()));  // empty append is always a no-op success
  EXPECT_FALSE(buf.append(std::string_view("x")));
  EXPECT_TRUE(buf.empty());
}

TEST(ByteBufferTest, CapacityFreedByConsumeCanBeReusedByLaterAppends) {
  ByteBuffer buf(/*max_capacity=*/4);
  ASSERT_TRUE(buf.append(std::string_view("abcd")));
  EXPECT_FALSE(buf.append(std::string_view("e")));  // full

  buf.consume(2);  // "ab" gone; "cd" remains, 2 bytes of headroom freed
  EXPECT_TRUE(buf.append(std::string_view("ef")));
  EXPECT_EQ(buf.view(), "cdef");
}

// Regression coverage for the compaction heuristic in
// compact_if_worthwhile(): many append/partial-consume cycles that net out
// to a steady-state logical size (well under the cap on every single
// cycle, but whose *cumulative* bytes ever appended is thousands of times
// the cap) must keep succeeding indefinitely — proving the underlying
// std::vector's allocated storage is genuinely being reclaimed as bytes
// are consumed, not silently growing forever underneath a size() that
// merely subtracts a never-reset read offset.
TEST(ByteBufferTest, ManyPartialConsumeCyclesAtSteadyStateNeverFail) {
  ByteBuffer buf(/*max_capacity=*/16);
  ASSERT_TRUE(buf.append(std::string_view("0123456789")));  // size 10
  buf.consume(4);                                            // size 6: a non-zero steady remainder
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(buf.append(std::string_view("0123456789")));  // size -> 16, exactly at the cap
    buf.consume(10);                                          // size -> 6 again; never fully drains,
    ASSERT_EQ(buf.size(), 6u);                                // so this exercises the *partial*
  }                                                            // compaction branch every iteration.
}

TEST(ByteBufferTest, MoveConstructionTransfersContent) {
  ByteBuffer buf(/*max_capacity=*/10);
  ASSERT_TRUE(buf.append(std::string_view("xyz")));

  ByteBuffer moved(std::move(buf));
  EXPECT_EQ(moved.view(), "xyz");
  EXPECT_EQ(moved.max_capacity(), 10u);
}

TEST(ByteBufferTest, AppendRawPointerOverload) {
  ByteBuffer buf;
  const char raw[] = {'a', 'b', 'c'};
  ASSERT_TRUE(buf.append(raw, sizeof(raw)));
  EXPECT_EQ(buf.view(), "abc");
}

}  // namespace
}  // namespace arcserve::buffers
