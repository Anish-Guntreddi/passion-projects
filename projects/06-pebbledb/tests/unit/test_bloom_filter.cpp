#include "pebbledb/bloom/bloom_filter.hpp"

#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// Pure in-memory build/query tests for the Bloom filter (spec FR3,
// roadmap Phase 5; ADR 0014). Real on-disk filter-block integration
// (sstable::Writer building one, sstable::Reader consulting it) lives in
// tests/integration/test_sstable_filter_and_cache.cpp.

namespace pebbledb::bloom {
namespace {

std::vector<std::string> MakeKeys(int count, const std::string& prefix = "key") {
  std::vector<std::string> keys;
  keys.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    keys.push_back(prefix + std::to_string(i));
  }
  return keys;
}

TEST(BloomFilter, EmptyKeysProducesEmptyFilter) {
  EXPECT_TRUE(BuildFilter({}, FilterPolicy{}).empty());
}

TEST(BloomFilter, ZeroBitsPerKeyDisablesFilterGeneration) {
  std::vector<std::string> keys = MakeKeys(10);
  EXPECT_TRUE(BuildFilter(keys, FilterPolicy{0}).empty());
}

// The core correctness contract (bloom_filter.hpp): no false negatives,
// ever. Every key actually added must report "may be present."
TEST(BloomFilter, NoFalseNegativesForAddedKeys) {
  std::vector<std::string> keys = MakeKeys(2000);
  std::string filter = BuildFilter(keys, FilterPolicy{10});
  ASSERT_FALSE(filter.empty());

  for (const std::string& key : keys) {
    EXPECT_TRUE(KeyMayMatch(key, filter)) << "key=" << key;
  }
}

// Not a correctness requirement (false positives are allowed by design),
// but a sanity check that the false-positive rate at the default 10
// bits/key is in the right ballpark (~1%, ADR 0014) rather than, say,
// 50% (which would indicate a broken hash/probe implementation, not
// merely "some false positives"). Generous bound (10%) to keep this
// robust against any single unlucky draw.
TEST(BloomFilter, FalsePositiveRateIsRoughlyInExpectedRange) {
  std::vector<std::string> keys = MakeKeys(5000, "present-");
  std::string filter = BuildFilter(keys, FilterPolicy{10});
  ASSERT_FALSE(filter.empty());

  int false_positives = 0;
  constexpr int kProbeCount = 20000;
  for (int i = 0; i < kProbeCount; ++i) {
    std::string absent_key = "absent-" + std::to_string(i);
    if (KeyMayMatch(absent_key, filter)) {
      ++false_positives;
    }
  }
  const double rate = static_cast<double>(false_positives) / kProbeCount;
  EXPECT_LT(rate, 0.10) << "false positive rate " << rate << " far exceeds the ~1% expected at 10 bits/key";
}

TEST(BloomFilter, KeyMayMatchOnEmptyFilterReturnsTrue) {
  // "No filter" (empty content) must degrade to "do the real lookup,"
  // never to "report absent" -- see bloom_filter.hpp.
  EXPECT_TRUE(KeyMayMatch("anything", std::string_view()));
}

TEST(BloomFilter, KeyMayMatchOnTooShortFilterReturnsTrue) {
  EXPECT_TRUE(KeyMayMatch("anything", std::string_view("\x05", 1)));  // 1 byte: no room for any bits
}

TEST(BloomFilter, DifferentKeySetsProduceDifferentFilters) {
  std::string filter_a = BuildFilter(MakeKeys(100, "a"), FilterPolicy{10});
  std::string filter_b = BuildFilter(MakeKeys(100, "b"), FilterPolicy{10});
  EXPECT_NE(filter_a, filter_b);
}

TEST(BloomFilter, SingleKeyRoundTrips) {
  std::vector<std::string> keys = {"only-key"};
  std::string filter = BuildFilter(keys, FilterPolicy{10});
  ASSERT_FALSE(filter.empty());
  EXPECT_TRUE(KeyMayMatch("only-key", filter));
}

TEST(BloomFilter, EmptyStringKeyIsValid) {
  std::vector<std::string> keys = {""};
  std::string filter = BuildFilter(keys, FilterPolicy{10});
  ASSERT_FALSE(filter.empty());
  EXPECT_TRUE(KeyMayMatch("", filter));
}

TEST(BloomFilter, BinarySafeKeysRoundTrip) {
  std::string key("k\0ey\xff", 5);
  std::vector<std::string> keys = {key};
  std::string filter = BuildFilter(keys, FilterPolicy{10});
  ASSERT_FALSE(filter.empty());
  EXPECT_TRUE(KeyMayMatch(key, filter));
}

}  // namespace
}  // namespace pebbledb::bloom
