#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pebbledb/cache/block_cache.hpp"
#include "pebbledb/sstable/reader.hpp"
#include "pebbledb/sstable/writer.hpp"
#include "temp_file.hpp"

// Roadmap Phase 5 exit criterion: "Benchmark demonstrates reduced
// unnecessary reads / cache effects." These tests measure that directly
// and deterministically via sstable::Reader's filter_checks()/
// filter_rejections()/cache_hits()/cache_misses()/data_blocks_read()
// counters (reader.hpp) -- real on-disk tables, real Bloom filter
// content (bloom/bloom_filter.hpp), real cache::BlockCache (ADR 0015) --
// rather than a wall-clock timing benchmark, which would be noisy and
// non-reproducible for a pass/fail test. Pure in-memory filter-building
// tests live in tests/unit/test_bloom_filter.cpp; pure in-memory cache
// tests in tests/unit/test_block_cache.cpp.

namespace pebbledb::sstable {
namespace {

using pebbledb::testutil::TempFile;

// Writes `count` sorted keys ("present-00000".."present-NNNNN") with
// small target_block_size_bytes so the table spans many data blocks --
// necessary for these tests to actually exercise per-block filtering/
// caching rather than everything landing in one block.
std::vector<std::string> WritePresentKeysTable(const std::string& path, int count,
                                                Writer::Options options,
                                                const std::string& value_tag = "value") {
  std::unique_ptr<Writer> writer;
  auto open_status = Writer::Open(path, options, &writer);
  EXPECT_TRUE(open_status.ok()) << open_status.ToString();

  std::vector<std::string> keys;
  keys.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "present-%05d", i);
    std::string key = buf;
    auto add_status =
        writer->Add(key, value_tag + "-" + std::to_string(i), static_cast<std::uint64_t>(i + 1),
                     EntryType::kValue);
    EXPECT_TRUE(add_status.ok()) << add_status.ToString();
    keys.push_back(std::move(key));
  }
  auto finish_status = writer->Finish();
  EXPECT_TRUE(finish_status.ok()) << finish_status.ToString();
  return keys;
}

// --- Bloom filter: reduced reads for absent keys --------------------------

TEST(SstableFilterAndCache, FilterAvoidsDataBlockReadsForMostAbsentKeys) {
  TempFile file;
  Writer::Options options;
  options.target_block_size_bytes = 64;  // force many small blocks
  options.filter_bits_per_key = 10;
  WritePresentKeysTable(file.path(), 500, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  ASSERT_TRUE(reader->has_filter());
  ASSERT_GT(reader->index_entry_count(), 10u) << "test needs many blocks to be meaningful";

  // "absent-*" sorts before every "present-*" key, so *without* a filter
  // the index search would still land on (and have to read) the first
  // data block for every one of these lookups -- see
  // WithoutFilterEveryAbsentLookupWithinRangeReadsADataBlock below for
  // that control case. With the filter, most of these should be
  // rejected before any data-block read happens at all.
  constexpr int kAbsentProbes = 300;
  for (int i = 0; i < kAbsentProbes; ++i) {
    const std::string absent_key = "absent-" + std::to_string(i);
    LookupResult result;
    std::string value;
    ASSERT_TRUE(reader->Get(absent_key, &result, &value).ok());
    EXPECT_EQ(result, LookupResult::kNotFound) << "key=" << absent_key;  // no false negatives
  }

  EXPECT_EQ(reader->filter_checks(), static_cast<std::uint64_t>(kAbsentProbes));
  // ~1% false-positive rate at the default 10 bits/key (ADR 0014) means
  // the overwhelming majority of these 300 probes should be rejected by
  // the filter; a generous 80% floor keeps this robust against any
  // single unlucky draw while still being a meaningful assertion.
  EXPECT_GT(reader->filter_rejections(), static_cast<std::uint64_t>(kAbsentProbes) * 8 / 10);
  EXPECT_LT(reader->data_blocks_read(), static_cast<std::uint64_t>(kAbsentProbes));
}

TEST(SstableFilterAndCache, WithoutFilterEveryAbsentLookupWithinRangeReadsADataBlock) {
  // Control for the test above: same table shape, filter disabled.
  TempFile file;
  Writer::Options options;
  options.target_block_size_bytes = 64;
  options.filter_bits_per_key = 0;
  WritePresentKeysTable(file.path(), 500, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());
  ASSERT_FALSE(reader->has_filter());

  constexpr int kAbsentProbes = 300;
  for (int i = 0; i < kAbsentProbes; ++i) {
    const std::string absent_key = "absent-" + std::to_string(i);
    LookupResult result;
    std::string value;
    ASSERT_TRUE(reader->Get(absent_key, &result, &value).ok());
    EXPECT_EQ(result, LookupResult::kNotFound);
  }

  EXPECT_EQ(reader->filter_checks(), 0u);
  // Every one of these keys sorts before the table's first (and every
  // other) key, so the index search lands on the first data block every
  // time -- with no filter to short-circuit it, that block is actually
  // read from disk on every single lookup.
  EXPECT_EQ(reader->data_blocks_read(), static_cast<std::uint64_t>(kAbsentProbes));
}

// --- Block cache: reduced reads for repeated lookups -----------------------

TEST(SstableFilterAndCache, CacheAvoidsRepeatedDataBlockReadsForSameKey) {
  TempFile file;
  Writer::Options options;
  options.target_block_size_bytes = 64;
  options.filter_bits_per_key = 0;  // isolate the cache's effect from the filter's
  std::vector<std::string> keys = WritePresentKeysTable(file.path(), 500, options);

  cache::BlockCache shared_cache(1024 * 1024);
  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader, &shared_cache, /*file_id=*/1).ok());

  const std::string& target_key = keys[250];
  constexpr int kRepeats = 20;
  for (int i = 0; i < kRepeats; ++i) {
    LookupResult result;
    std::string value;
    ASSERT_TRUE(reader->Get(target_key, &result, &value).ok());
    EXPECT_EQ(result, LookupResult::kFound);
  }

  EXPECT_EQ(reader->cache_misses(), 1u);
  EXPECT_EQ(reader->cache_hits(), static_cast<std::uint64_t>(kRepeats - 1));
  // Only the very first lookup actually touched disk for this block --
  // the roadmap Phase 5 exit criterion, demonstrated directly.
  EXPECT_EQ(reader->data_blocks_read(), 1u);

  cache::BlockCache::Stats stats = shared_cache.GetStats();
  EXPECT_EQ(stats.hits, static_cast<std::uint64_t>(kRepeats - 1));
  EXPECT_EQ(stats.misses, 1u);
}

TEST(SstableFilterAndCache, WithoutCacheEveryLookupReadsADataBlock) {
  // Control for the test above: no cache supplied to Open().
  TempFile file;
  Writer::Options options;
  options.target_block_size_bytes = 64;
  options.filter_bits_per_key = 0;
  std::vector<std::string> keys = WritePresentKeysTable(file.path(), 500, options);

  std::unique_ptr<Reader> reader;
  ASSERT_TRUE(Reader::Open(file.path(), &reader).ok());  // no cache

  const std::string& target_key = keys[250];
  constexpr int kRepeats = 20;
  for (int i = 0; i < kRepeats; ++i) {
    LookupResult result;
    std::string value;
    ASSERT_TRUE(reader->Get(target_key, &result, &value).ok());
  }

  EXPECT_EQ(reader->data_blocks_read(), static_cast<std::uint64_t>(kRepeats));
  EXPECT_EQ(reader->cache_hits(), 0u);
  EXPECT_EQ(reader->cache_misses(), 0u);
}

// D8's key shape, end to end: (file_id, block_offset). Two different
// tables sharing one BlockCache, each with a key at the very same
// on-disk block offset (both tables' first block, since both are built
// the same way), must never serve one table's cached bytes to the
// other.
TEST(SstableFilterAndCache, SharedCacheAcrossTwoTablesDoesNotCollideOnOffset) {
  TempFile file_a;
  TempFile file_b;
  Writer::Options options;
  options.target_block_size_bytes = 64;
  options.filter_bits_per_key = 0;
  std::vector<std::string> keys_a = WritePresentKeysTable(file_a.path(), 200, options, "table-a");
  std::vector<std::string> keys_b = WritePresentKeysTable(file_b.path(), 200, options, "table-b");
  ASSERT_EQ(keys_a[0], keys_b[0]);  // same key text in both tables, by construction

  cache::BlockCache shared_cache(1024 * 1024);
  std::unique_ptr<Reader> reader_a;
  std::unique_ptr<Reader> reader_b;
  ASSERT_TRUE(Reader::Open(file_a.path(), &reader_a, &shared_cache, /*file_id=*/1).ok());
  ASSERT_TRUE(Reader::Open(file_b.path(), &reader_b, &shared_cache, /*file_id=*/2).ok());

  LookupResult result_a;
  std::string value_a;
  ASSERT_TRUE(reader_a->Get(keys_a[0], &result_a, &value_a).ok());
  ASSERT_EQ(result_a, LookupResult::kFound);
  EXPECT_EQ(value_a, "table-a-0");

  LookupResult result_b;
  std::string value_b;
  ASSERT_TRUE(reader_b->Get(keys_b[0], &result_b, &value_b).ok());
  ASSERT_EQ(result_b, LookupResult::kFound);
  EXPECT_EQ(value_b, "table-b-0");  // not table-a's cached value at the same offset

  // Both were genuine cache misses (distinct file_id keys), and each
  // Reader independently caused exactly one data-block read.
  EXPECT_EQ(reader_a->cache_misses(), 1u);
  EXPECT_EQ(reader_b->cache_misses(), 1u);
  EXPECT_EQ(reader_a->data_blocks_read(), 1u);
  EXPECT_EQ(reader_b->data_blocks_read(), 1u);
}

}  // namespace
}  // namespace pebbledb::sstable
