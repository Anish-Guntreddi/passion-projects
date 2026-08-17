#include "pebbledb/db.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// Regression coverage for the concurrent-Get() data race fixed per ADR
// 0003 (docs/decisions/0003-concurrency-model.md): Get() only takes DB's
// *shared* lock (by design, so concurrent reads don't serialize on each
// other), so its per-call counters (get_count/get_hit_count/
// get_miss_count) must be atomics, not plain integers merely "protected"
// by holding a shared lock together with other concurrent readers.
// Without that fix, this file fails under ThreadSanitizer
// (build/Debug-tsan, PEBBLEDB_ENABLE_TSAN) with a data race report on
// DB::stats_, and can silently undercount even without TSan, depending on
// scheduling.

namespace pebbledb {
namespace {

TEST(DbConcurrency, ConcurrentGetCallsAreRaceFreeAndCountEveryCall) {
  DB db;
  ASSERT_TRUE(db.Put("k1", "v1").ok());

  constexpr int kThreads = 8;
  constexpr int kGetsPerThread = 500;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&db, t]() {
      for (int i = 0; i < kGetsPerThread; ++i) {
        std::string value;
        // Alternate between a hit and a miss so get_hit_count and
        // get_miss_count are both exercised concurrently, not just the
        // overall get_count.
        if ((t + i) % 2 == 0) {
          EXPECT_TRUE(db.Get("k1", &value).ok());
        } else {
          EXPECT_TRUE(db.Get("missing", &value).IsNotFound());
        }
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }

  DB::Stats stats = db.GetStats();
  constexpr auto kExpectedTotal =
      static_cast<std::uint64_t>(kThreads) * static_cast<std::uint64_t>(kGetsPerThread);
  // If any increment were lost to the race this test guards against, this
  // would undercount rather than fail an ASan/UBSan check -- exact
  // equality is the point: every one of the kExpectedTotal Get() calls
  // above must be reflected exactly once.
  EXPECT_EQ(stats.get_count, kExpectedTotal);
  EXPECT_EQ(stats.get_hit_count + stats.get_miss_count, kExpectedTotal);
}

// Broader smoke coverage for ADR 0003's concurrency contract: concurrent
// readers running continuously while a single writer mutates the map must
// never crash or trip a sanitizer, regardless of interleaving. This does
// not assert on specific counts (the writer and readers race by design;
// only the *absence of UB* is being verified here) -- ThreadSanitizer is
// what actually validates "no data race" for this test.
TEST(DbConcurrency, ConcurrentReadersDuringWritesStayMemorySafe) {
  DB db;
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(db.Put("key" + std::to_string(i), "v").ok());
  }

  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&db, &stop]() {
      while (!stop.load(std::memory_order_relaxed)) {
        std::string value;
        (void)db.Get("key0", &value);
        (void)db.GetStats();
      }
    });
  }

  std::thread writer([&db, &stop]() {
    for (int i = 0; i < 2000; ++i) {
      (void)db.Put("key" + std::to_string(i % 10), "v" + std::to_string(i));
    }
    stop.store(true, std::memory_order_relaxed);
  });

  writer.join();
  for (std::thread& th : readers) {
    th.join();
  }
}

}  // namespace
}  // namespace pebbledb
