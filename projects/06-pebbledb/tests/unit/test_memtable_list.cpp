#include "pebbledb/memtable/memtable_list.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

// Phase 2 exit criterion (spec roadmap): "Reads merge active state
// correctly; flush can snapshot immutable state." These tests exercise
// memtable::MemTableList, which owns the active + immutable MemTable
// stack and implements the newest-layer-wins read merge across it.

namespace pebbledb::memtable {
namespace {

TEST(MemTableList, StartsWithOneEmptyActiveTableAndNoImmutables) {
  MemTableList list;
  EXPECT_EQ(list.ImmutableCount(), 0u);
  EXPECT_TRUE(list.active().empty());
}

TEST(MemTableList, PutAndGetWorkDirectlyOnActiveTable) {
  MemTableList list;
  list.Put("k1", "v1", 1);

  std::string value;
  ASSERT_EQ(list.Get("k1", &value), LookupResult::kFound);
  EXPECT_EQ(value, "v1");
}

TEST(MemTableList, DeleteOnActiveTableIsVisibleAsTombstone) {
  MemTableList list;
  list.Put("k1", "v1", 1);
  list.Delete("k1", 2);

  std::string value;
  EXPECT_EQ(list.Get("k1", &value), LookupResult::kDeleted);
}

TEST(MemTableList, GetOnNeverSeenKeyIsNotFound) {
  MemTableList list;
  std::string value;
  EXPECT_EQ(list.Get("missing", &value), LookupResult::kNotFound);
}

// Phase 2 exit criterion, "flush can snapshot immutable state": Freeze()
// must produce a stable view that further writes to the (new) active
// table cannot mutate.
TEST(MemTableList, FreezeMovesActiveToImmutableAndInstallsFreshActive) {
  MemTableList list;
  list.Put("k1", "v1", 1);

  std::shared_ptr<MemTable> frozen = list.Freeze();
  ASSERT_NE(frozen, nullptr);
  EXPECT_EQ(frozen->entry_count(), 1u);

  EXPECT_TRUE(list.active().empty());
  EXPECT_EQ(list.ImmutableCount(), 1u);
}

TEST(MemTableList, FrozenSnapshotIsUnaffectedByFurtherWritesToNewActiveTable) {
  MemTableList list;
  list.Put("k1", "v1", 1);
  std::shared_ptr<MemTable> frozen = list.Freeze();

  // Further writes go to the brand-new active table, not the frozen one.
  list.Put("k1", "v1-updated-in-new-active", 2);
  list.Put("k2", "v2", 3);

  std::string value;
  ASSERT_EQ(frozen->Get("k1", &value), LookupResult::kFound);
  EXPECT_EQ(value, "v1");  // unchanged
  EXPECT_EQ(frozen->entry_count(), 1u);  // k2 never reached the frozen table
}

// Core merge rule (docs/architecture.md read path): active wins outright
// over anything in an immutable table beneath it.
TEST(MemTableList, GetPrefersActiveOverImmutableForSameKey) {
  MemTableList list;
  list.Put("k1", "old-in-immutable", 1);
  list.Freeze();
  list.Put("k1", "new-in-active", 2);

  std::string value;
  ASSERT_EQ(list.Get("k1", &value), LookupResult::kFound);
  EXPECT_EQ(value, "new-in-active");
}

// Invariant 4 (spec §1.5) at the merge level: a tombstone in the active
// table must shadow a live value in an immutable table beneath it -- the
// old value must never resurface.
TEST(MemTableList, Invariant4_TombstoneInActiveShadowsOlderValueInImmutable) {
  MemTableList list;
  list.Put("k1", "old-value", 1);
  list.Freeze();
  list.Delete("k1", 2);

  std::string value;
  EXPECT_EQ(list.Get("k1", &value), LookupResult::kDeleted);
}

// The other half of invariant 4: a tombstone recorded only in an
// immutable table (nothing in the active table for that key at all) must
// still be honored -- Get() must not skip past it to search SSTables
// (Phase 4+) or, in this pre-SSTable phase, must not report kNotFound.
TEST(MemTableList, Invariant4_TombstoneInImmutableIsVisibleWhenActiveHasNothing) {
  MemTableList list;
  list.Put("k1", "v1", 1);
  list.Delete("k1", 2);
  list.Freeze();
  // New active table has never heard of "k1" at all.

  std::string value;
  EXPECT_EQ(list.Get("k1", &value), LookupResult::kDeleted);
}

TEST(MemTableList, MultipleImmutablesAreSearchedNewestFirst) {
  MemTableList list;
  list.Put("k1", "oldest", 1);
  list.Freeze();  // immutables: [oldest]
  list.Put("k1", "middle", 2);
  list.Freeze();  // immutables: [middle, oldest]
  // Active table has nothing for k1.

  ASSERT_EQ(list.ImmutableCount(), 2u);
  std::string value;
  ASSERT_EQ(list.Get("k1", &value), LookupResult::kFound);
  EXPECT_EQ(value, "middle");  // the newer immutable wins, not the oldest
}

TEST(MemTableList, KeyAbsentFromEveryLayerIsNotFound) {
  MemTableList list;
  list.Put("k1", "v1", 1);
  list.Freeze();
  list.Put("k2", "v2", 2);

  std::string value;
  EXPECT_EQ(list.Get("k3", &value), LookupResult::kNotFound);
}

TEST(MemTableList, ImmutablesAccessorIsOrderedNewestFirst) {
  MemTableList list;
  list.Put("a", "first-frozen", 1);
  std::shared_ptr<MemTable> first = list.Freeze();
  list.Put("b", "second-frozen", 2);
  std::shared_ptr<MemTable> second = list.Freeze();

  const auto& immutables = list.immutables();
  ASSERT_EQ(immutables.size(), 2u);
  EXPECT_EQ(immutables[0], second);  // most recently frozen is first
  EXPECT_EQ(immutables[1], first);
}

TEST(MemTableList, ActiveShouldFlushRespectsThreshold) {
  MemTableList list;
  EXPECT_FALSE(list.ActiveShouldFlush(1024));

  list.Put("k1", std::string(2000, 'x'), 1);
  EXPECT_FALSE(list.ActiveShouldFlush(1024 * 1024));  // well under a 1 MiB threshold
  EXPECT_TRUE(list.ActiveShouldFlush(100));            // well over a 100-byte threshold
}

TEST(MemTableList, ActiveApproximateMemoryUsageMatchesActiveTable) {
  MemTableList list;
  list.Put("k1", "v1", 1);
  EXPECT_EQ(list.ActiveApproximateMemoryUsage(), list.active().ApproximateMemoryUsage());
}

}  // namespace
}  // namespace pebbledb::memtable
