#include "arcserve/reactor/output_queue.hpp"

#include <algorithm>
#include <utility>

namespace arcserve::reactor {

bool OutputQueue::enqueue(std::string_view data) {
  if (data.empty()) {
    return true;  // No-op; trivially within any bound.
  }
  if (data.size() > max_buffered_bytes_ - std::min(queued_bytes_, max_buffered_bytes_)) {
    return false;  // Would push queued_bytes() past the cap.
  }

  // Each chunk gets its own exactly-sized cap (== data.size()), so its
  // append() below is guaranteed to succeed — the aggregate bound was
  // already checked above; ByteBuffer's own cap here is just what makes
  // this one chunk itself a valid, self-contained ByteBuffer, not a second
  // independent limit.
  buffers::ByteBuffer chunk(data.size());
  [[maybe_unused]] bool appended = chunk.append(data);
  chunks_.push_back(std::move(chunk));
  queued_bytes_ += data.size();
  return true;
}

std::string_view OutputQueue::front() const noexcept {
  if (chunks_.empty()) {
    return {};
  }
  return chunks_.front().view();
}

void OutputQueue::consume(std::size_t n) noexcept {
  if (chunks_.empty()) {
    return;
  }
  std::size_t taken = std::min(n, chunks_.front().size());
  chunks_.front().consume(taken);
  queued_bytes_ -= taken;
  if (chunks_.front().empty()) {
    chunks_.pop_front();
  }
}

}  // namespace arcserve::reactor
