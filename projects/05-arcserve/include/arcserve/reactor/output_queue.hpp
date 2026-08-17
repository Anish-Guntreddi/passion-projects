#pragma once

#include <cstddef>
#include <deque>
#include <string_view>

#include "arcserve/buffers/byte_buffer.hpp"

namespace arcserve::reactor {

// Default cap on aggregate queued-but-unsent output bytes per connection —
// generous enough that no existing correctness test in this codebase (the
// largest response body used anywhere is HttpRequestParser::kMaxBodyBytes,
// 1 MiB) comes close to it, while still being a real, finite bound: see
// docs/decisions/0006-buffer-representation.md.
inline constexpr std::size_t kDefaultMaxOutputQueueBytes = 16 * 1024 * 1024;  // 16 MiB

// A bounded, multi-chunk send queue — Phase 4's "write queues" roadmap
// item, and the formalization docs/decisions/0005-reactor-connection-
// model.md's decision 2 predicted: each enqueue() call adds one discrete
// chunk (typically one serialized HttpResponse, or one echoed read for
// EchoReactorServer) behind whatever is already queued; the *aggregate*
// unconsumed byte count across every still-pending chunk is capped at
// max_buffered_bytes(), enforced by ByteBuffer's own all-or-nothing
// append() bound underneath (see docs/decisions/0006).
//
// This class is deliberately NOT the general overload/high-water-mark
// *policy* machinery FR5/Phase 6 describe (soft marks, pause-vs-reject,
// observable rejection counters) — it enforces exactly one hard cap and
// reports true/false; what a caller does with a false return (every server
// in this codebase today: close the connection) is Phase 6's to make
// configurable and observable.
//
// Not thread-safe; not copyable or movable (a Connection owns exactly one
// OutputQueue for its whole lifetime, embedded by value — nothing ever
// needs to relocate one). One instance belongs to one Connection, touched
// only from the reactor thread that owns it.
class OutputQueue {
 public:
  explicit OutputQueue(std::size_t max_buffered_bytes = kDefaultMaxOutputQueueBytes) noexcept
      : max_buffered_bytes_(max_buffered_bytes) {}

  OutputQueue(const OutputQueue&) = delete;
  OutputQueue& operator=(const OutputQueue&) = delete;
  OutputQueue(OutputQueue&&) = delete;
  OutputQueue& operator=(OutputQueue&&) = delete;
  ~OutputQueue() = default;

  // Enqueues `data` as a new chunk behind whatever is already queued.
  // Returns false (enqueuing nothing) if doing so would push queued_bytes()
  // past max_buffered_bytes() — an all-or-nothing bounded push, exactly
  // like ByteBuffer::append() (which is what this delegates the actual
  // bound check to, one chunk at a time).
  [[nodiscard]] bool enqueue(std::string_view data);

  // The unconsumed bytes of the oldest (front) chunk — contiguous, ready to
  // hand straight to send(2)/write_all(). Empty if the queue itself is
  // empty. Never spans more than one chunk: a caller that needs to drain
  // across a chunk boundary calls consume() to retire the front chunk once
  // it's fully sent, then calls front() again for the next one — see
  // Connection::flush_output() for the loop that does exactly this.
  [[nodiscard]] std::string_view front() const noexcept;

  // Marks `n` bytes of the front chunk as sent (n is clamped to
  // front().size(), same forgiving contract as ByteBuffer::consume()),
  // popping that chunk once it's fully drained. A no-op if the queue is
  // already empty.
  void consume(std::size_t n) noexcept;

  [[nodiscard]] bool empty() const noexcept { return chunks_.empty(); }
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return queued_bytes_; }
  [[nodiscard]] std::size_t queued_chunks() const noexcept { return chunks_.size(); }
  [[nodiscard]] std::size_t max_buffered_bytes() const noexcept { return max_buffered_bytes_; }

 private:
  std::deque<buffers::ByteBuffer> chunks_;
  std::size_t queued_bytes_ = 0;
  std::size_t max_buffered_bytes_;
};

}  // namespace arcserve::reactor
