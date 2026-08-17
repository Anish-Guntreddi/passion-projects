#include "pebbledb/memtable/memtable.hpp"

#include <utility>

namespace pebbledb::memtable {

namespace {
std::size_t EntryCost(std::size_t key_size, std::size_t value_size) {
  return key_size + value_size + MemTable::kPerEntryOverheadBytes;
}
}  // namespace

void MemTable::Put(Slice key, Slice value, std::uint64_t sequence) {
  Entry entry;
  entry.sequence = sequence;
  entry.type = EntryType::kValue;
  entry.value.assign(value.data(), value.size());

  const std::size_t new_cost = EntryCost(key.size(), value.size());
  auto it = table_.find(key);
  if (it != table_.end()) {
    approximate_memory_usage_ -= EntryCost(it->first.size(), it->second.value.size());
    approximate_memory_usage_ += new_cost;
    it->second = std::move(entry);
  } else {
    approximate_memory_usage_ += new_cost;
    table_.emplace(std::string(key), std::move(entry));
  }
}

void MemTable::Delete(Slice key, std::uint64_t sequence) {
  Entry entry;
  entry.sequence = sequence;
  entry.type = EntryType::kTombstone;
  // entry.value stays empty -- a tombstone never carries a value.

  const std::size_t new_cost = EntryCost(key.size(), 0);
  auto it = table_.find(key);
  if (it != table_.end()) {
    approximate_memory_usage_ -= EntryCost(it->first.size(), it->second.value.size());
    approximate_memory_usage_ += new_cost;
    it->second = std::move(entry);
  } else {
    approximate_memory_usage_ += new_cost;
    table_.emplace(std::string(key), std::move(entry));
  }
}

LookupResult MemTable::Get(Slice key, std::string* value, std::uint64_t* sequence) const {
  auto it = table_.find(key);
  if (it == table_.end()) {
    return LookupResult::kNotFound;
  }
  if (sequence != nullptr) {
    *sequence = it->second.sequence;
  }
  if (it->second.type == EntryType::kTombstone) {
    return LookupResult::kDeleted;
  }
  *value = it->second.value;
  return LookupResult::kFound;
}

}  // namespace pebbledb::memtable
