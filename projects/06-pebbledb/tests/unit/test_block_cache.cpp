#include "pebbledb/cache/block_cache.hpp"

#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// BlockCache unit tests (spec FR7, roadmap Phase 5; decision D8; ADR
// 0015). Real sstable::Reader integration (actual data blocks flowing
// through this cache) lives in
// tests/integration/test_sstable_filter_and_cache.cpp.

namespace pebbledb::cache {
namespace {

TEST(BlockCache, LookupOnEmptyCacheMisses) {
  BlockCache cache(1024);
  std::string out;
  EXPECT_FALSE(cache.Lookup(1, 0, &out));
  EXPECT_EQ(cache.GetStats().misses, 1u);
}

TEST(BlockCache, InsertThenLookupHits) {
  BlockCache cache(1024);
  cache.Insert(1, 100, "block-contents");

  std::string out;
  ASSERT_TRUE(cache.Lookup(1, 100, &out));
  EXPECT_EQ(out, "block-contents");

  BlockCache::Stats stats = cache.GetStats();
  EXPECT_EQ(stats.hits, 1u);
  EXPECT_EQ(stats.misses, 0u);
  EXPECT_EQ(stats.inserts, 1u);
  EXPECT_EQ(stats.current_bytes, std::string("block-contents").size());
}

// D8's key shape: (file_id, block_offset) -- two different files with
// blocks at the same offset must not collide.
TEST(BlockCache, DifferentFileIdsWithSameOffsetDoNotCollide) {
  BlockCache cache(1024);
  cache.Insert(1, 0, "file-one-block");
  cache.Insert(2, 0, "file-two-block");

  std::string out;
  ASSERT_TRUE(cache.Lookup(1, 0, &out));
  EXPECT_EQ(out, "file-one-block");
  ASSERT_TRUE(cache.Lookup(2, 0, &out));
  EXPECT_EQ(out, "file-two-block");
}

TEST(BlockCache, InsertOverwritesExistingEntry) {
  BlockCache cache(1024);
  cache.Insert(1, 0, "old-contents");
  cache.Insert(1, 0, "new-contents-longer");

  std::string out;
  ASSERT_TRUE(cache.Lookup(1, 0, &out));
  EXPECT_EQ(out, "new-contents-longer");
  EXPECT_EQ(cache.GetStats().current_bytes, std::string("new-contents-longer").size());
}

// Core LRU behavior: with capacity for exactly two 10-byte blocks,
// inserting a third evicts the *least recently used* one, not simply the
// oldest-inserted one -- Lookup()'s promote-to-front must actually take
// effect.
TEST(BlockCache, EvictsLeastRecentlyUsedNotOldestInserted) {
  BlockCache cache(20);  // room for exactly two 10-byte blocks
  cache.Insert(1, 0, std::string(10, 'a'));
  cache.Insert(1, 1, std::string(10, 'b'));

  // Touch block 0 so block 1 becomes the least-recently-used one.
  std::string out;
  ASSERT_TRUE(cache.Lookup(1, 0, &out));

  cache.Insert(1, 2, std::string(10, 'c'));  // forces one eviction

  EXPECT_TRUE(cache.Lookup(1, 0, &out));   // recently touched -- still present
  EXPECT_FALSE(cache.Lookup(1, 1, &out));  // least-recently-used -- evicted
  EXPECT_TRUE(cache.Lookup(1, 2, &out));   // just inserted -- present

  EXPECT_EQ(cache.GetStats().evictions, 1u);
  EXPECT_LE(cache.GetStats().current_bytes, 20u);
}

TEST(BlockCache, CapacityIsNeverExceededAcrossManyInserts) {
  BlockCache cache(100);
  for (std::uint64_t i = 0; i < 50; ++i) {
    cache.Insert(1, i, std::string(10, 'x'));
    EXPECT_LE(cache.GetStats().current_bytes, 100u);
  }
  EXPECT_GT(cache.GetStats().evictions, 0u);
}

// A single block larger than the whole cache capacity is still
// inserted, not rejected -- see BlockCache::Insert's doc comment -- but
// the cache's "never exceed capacity_bytes" invariant then evicts it
// again immediately (along with everything else), since it alone cannot
// fit. Net effect: the cache ends up empty, not "over capacity."
TEST(BlockCache, OversizedSingleBlockIsInsertedThenImmediatelyEvicted) {
  BlockCache cache(10);
  cache.Insert(1, 0, std::string(5, 'a'));
  cache.Insert(1, 1, std::string(50, 'b'));  // far exceeds capacity on its own

  std::string out;
  EXPECT_FALSE(cache.Lookup(1, 0, &out));  // evicted to make room
  EXPECT_FALSE(cache.Lookup(1, 1, &out));  // also evicted -- alone exceeds capacity
  EXPECT_EQ(cache.GetStats().current_bytes, 0u);
  EXPECT_EQ(cache.GetStats().evictions, 2u);
}

TEST(BlockCache, ZeroCapacityCacheAlwaysMisses) {
  BlockCache cache(0);
  cache.Insert(1, 0, "some-block");

  std::string out;
  EXPECT_FALSE(cache.Lookup(1, 0, &out));
  EXPECT_EQ(cache.GetStats().current_bytes, 0u);
}

TEST(BlockCache, EraseDropsOnlyMatchingFileId) {
  BlockCache cache(1024);
  cache.Insert(1, 0, "a");
  cache.Insert(1, 1, "b");
  cache.Insert(2, 0, "c");

  cache.Erase(1);

  std::string out;
  EXPECT_FALSE(cache.Lookup(1, 0, &out));
  EXPECT_FALSE(cache.Lookup(1, 1, &out));
  EXPECT_TRUE(cache.Lookup(2, 0, &out));
}

TEST(BlockCache, EraseOnAbsentFileIdIsNoOp) {
  BlockCache cache(1024);
  cache.Insert(1, 0, "a");
  cache.Erase(999);

  std::string out;
  EXPECT_TRUE(cache.Lookup(1, 0, &out));
}

// Concurrency smoke test: many threads racing Lookup()/Insert() across a
// small key space must never crash or trip a sanitizer (ThreadSanitizer
// is what actually validates "no data race" here -- see
// scripts/test.sh's tsan variant). Mirrors
// tests/unit/test_db_concurrency.cpp's
// ConcurrentReadersDuringWritesStayMemorySafe in spirit: no assertions
// on exact hit/miss counts (the threads race by design), only on memory
// safety.
TEST(BlockCache, ConcurrentLookupAndInsertStayMemorySafe) {
  BlockCache cache(4096);
  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 500;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&cache, t]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        const auto offset = static_cast<std::uint64_t>((t + i) % 16);
        if (i % 2 == 0) {
          cache.Insert(1, offset, std::string(8, 'x'));
        } else {
          std::string out;
          (void)cache.Lookup(1, offset, &out);
        }
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }
  // No crash, no sanitizer report -- that is the entire assertion.
}

}  // namespace
}  // namespace pebbledb::cache
