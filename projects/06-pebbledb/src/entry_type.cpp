#include "pebbledb/entry_type.hpp"

namespace pebbledb {

const char* EntryTypeName(EntryType type) {
  switch (type) {
    case EntryType::kValue:
      return "Value";
    case EntryType::kTombstone:
      return "Tombstone";
  }
  return "Unknown";
}

}  // namespace pebbledb
