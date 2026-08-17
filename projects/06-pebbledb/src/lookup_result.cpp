#include "pebbledb/lookup_result.hpp"

namespace pebbledb {

const char* LookupResultName(LookupResult result) {
  switch (result) {
    case LookupResult::kFound:
      return "Found";
    case LookupResult::kDeleted:
      return "Deleted";
    case LookupResult::kNotFound:
      return "NotFound";
  }
  return "Unknown";
}

}  // namespace pebbledb
