#include "arcserve/buffers/byte_buffer.hpp"

#include <algorithm>
#include <cstring>

namespace arcserve::buffers {

bool ByteBuffer::append(const void* data, std::size_t length) {
  if (length == 0) {
    return true;  // Trivially within any capacity, including zero.
  }
  // size() <= max_capacity_ is an invariant maintained by every prior
  // append(), so max_capacity_ - size() below can never underflow; written
  // this way (rather than `size() + length > max_capacity_`) specifically
  // so it can't overflow either, for any size_t value of length.
  if (length > max_capacity_ - size()) {
    return false;  // Would exceed the cap — all-or-nothing, unchanged.
  }

  std::size_t old_size = storage_.size();
  storage_.resize(old_size + length);
  std::memcpy(storage_.data() + old_size, data, length);
  return true;
}

void ByteBuffer::consume(std::size_t n) noexcept {
  std::size_t taken = std::min(n, size());
  read_offset_ += taken;
  compact_if_worthwhile();
}

void ByteBuffer::clear() noexcept {
  storage_.clear();
  read_offset_ = 0;
}

void ByteBuffer::compact_if_worthwhile() {
  if (read_offset_ == storage_.size()) {
    // Fully drained — the overwhelmingly common case (see class docs).
    // storage_.clear() keeps whatever capacity was already allocated for
    // the next append(), rather than needing a fresh allocation.
    storage_.clear();
    read_offset_ = 0;
    return;
  }
  if (read_offset_ > 0 && read_offset_ >= storage_.size() / 2) {
    // More than half consumed but not fully drained: worth paying an
    // O(remaining) shift now to reclaim the consumed prefix's space,
    // rather than letting storage_ grow unboundedly across many small
    // partial-consume cycles on a buffer that's still logically capped.
    storage_.erase(storage_.begin(), storage_.begin() + static_cast<std::ptrdiff_t>(read_offset_));
    read_offset_ = 0;
  }
}

}  // namespace arcserve::buffers
