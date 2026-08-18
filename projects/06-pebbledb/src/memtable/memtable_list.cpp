#include "pebbledb/memtable/memtable_list.hpp"

namespace pebbledb::memtable {

MemTableList::MemTableList() : active_(std::make_shared<MemTable>()) {}

void MemTableList::Put(Slice key, Slice value, std::uint64_t sequence) {
  active_->Put(key, value, sequence);
}

void MemTableList::Delete(Slice key, std::uint64_t sequence) {
  active_->Delete(key, sequence);
}

LookupResult MemTableList::Get(Slice key, std::string* value, std::uint64_t* sequence) const {
  LookupResult result = active_->Get(key, value, sequence);
  if (result != LookupResult::kNotFound) {
    return result;
  }
  for (const auto& immutable : immutables_) {
    result = immutable->Get(key, value, sequence);
    if (result != LookupResult::kNotFound) {
      return result;
    }
  }
  return LookupResult::kNotFound;
}

std::shared_ptr<MemTable> MemTableList::Freeze() {
  std::shared_ptr<MemTable> frozen = active_;
  immutables_.push_front(frozen);
  active_ = std::make_shared<MemTable>();
  return frozen;
}

void MemTableList::RemoveImmutable(const std::shared_ptr<MemTable>& table) {
  for (auto it = immutables_.begin(); it != immutables_.end(); ++it) {
    if (*it == table) {
      immutables_.erase(it);
      return;
    }
  }
}

}  // namespace pebbledb::memtable
