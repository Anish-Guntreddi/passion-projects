#include "pebbledb/db.hpp"

#include <mutex>
#include <shared_mutex>

namespace pebbledb {

Status DB::Put(Slice key, Slice value) {
  std::unique_lock lock(mutex_);
  auto [it, inserted] = table_.insert_or_assign(std::string(key), std::string(value));
  (void)it;
  ++stats_.put_count;
  if (inserted) {
    ++stats_.live_key_count;
  }
  return Status::OK();
}

Status DB::Delete(Slice key) {
  std::unique_lock lock(mutex_);
  ++stats_.delete_count;
  auto it = table_.find(key);
  if (it != table_.end()) {
    table_.erase(it);
    --stats_.live_key_count;
  }
  // Deleting an absent key is idempotent, not an error — see db.hpp.
  return Status::OK();
}

Status DB::Get(Slice key, std::string* value) const {
  std::shared_lock lock(mutex_);
  // Multiple Get() calls can hold this shared lock at the same instant
  // (that's the point of a shared lock), so these counters must be
  // updated atomically rather than with plain ++ — see db.hpp / ADR 0003.
  get_count_.fetch_add(1, std::memory_order_relaxed);
  auto it = table_.find(key);
  if (it == table_.end()) {
    get_miss_count_.fetch_add(1, std::memory_order_relaxed);
    return Status::NotFound();
  }
  get_hit_count_.fetch_add(1, std::memory_order_relaxed);
  *value = it->second;
  return Status::OK();
}

Status DB::Flush() { return Status::OK(); }

Status DB::Compact() { return Status::OK(); }

DB::Stats DB::GetStats() const {
  std::shared_lock lock(mutex_);
  Stats snapshot = stats_;
  snapshot.get_count = get_count_.load(std::memory_order_relaxed);
  snapshot.get_hit_count = get_hit_count_.load(std::memory_order_relaxed);
  snapshot.get_miss_count = get_miss_count_.load(std::memory_order_relaxed);
  return snapshot;
}

}  // namespace pebbledb
