#include "pebbledb/cache/block_cache.hpp"

#include <cassert>

namespace pebbledb::cache {

BlockCache::BlockCache(std::size_t capacity_bytes) : capacity_bytes_(capacity_bytes) {}

bool BlockCache::Lookup(std::uint64_t file_id, std::uint64_t block_offset, std::string* out) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = index_.find(Key{file_id, block_offset});
  if (it == index_.end()) {
    ++stats_.misses;
    return false;
  }
  // Promote to most-recently-used: splice the entry's node to the front
  // of lru_ in O(1) (splice moves the node itself, no copy/allocation,
  // and does not invalidate the iterator index_ still holds for it).
  lru_.splice(lru_.begin(), lru_, it->second);
  *out = it->second->second;
  ++stats_.hits;
  return true;
}

void BlockCache::Insert(std::uint64_t file_id, std::uint64_t block_offset, std::string block) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Key key{file_id, block_offset};

  auto existing = index_.find(key);
  if (existing != index_.end()) {
    // Overwrite in place: adjust current_bytes_ for the size delta, move
    // the entry to most-recently-used, replace its bytes.
    current_bytes_ -= existing->second->second.size();
    current_bytes_ += block.size();
    lru_.splice(lru_.begin(), lru_, existing->second);
    existing->second->second = std::move(block);
  } else {
    lru_.emplace_front(key, std::move(block));
    current_bytes_ += lru_.front().second.size();
    index_.emplace(key, lru_.begin());
  }
  ++stats_.inserts;

  EvictToCapacityLocked();
}

void BlockCache::EvictToCapacityLocked() {
  while (current_bytes_ > capacity_bytes_ && !lru_.empty()) {
    auto& [evict_key, evict_bytes] = lru_.back();
    current_bytes_ -= evict_bytes.size();
    index_.erase(evict_key);
    lru_.pop_back();
    ++stats_.evictions;
  }
}

void BlockCache::Erase(std::uint64_t file_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = lru_.begin(); it != lru_.end();) {
    if (it->first.file_id == file_id) {
      current_bytes_ -= it->second.size();
      index_.erase(it->first);
      it = lru_.erase(it);
    } else {
      ++it;
    }
  }
}

BlockCache::Stats BlockCache::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Stats snapshot = stats_;
  snapshot.current_bytes = current_bytes_;
  return snapshot;
}

}  // namespace pebbledb::cache
