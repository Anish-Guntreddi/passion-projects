#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace arcserve::buffers {

// Sentinel meaning "no capacity limit". Used sparingly and deliberately —
// every buffer actually sitting on the reactor's I/O path (Connection::
// read_buffer, every chunk inside a reactor::OutputQueue) is constructed
// with a real, finite cap; see docs/decisions/0006-buffer-representation.md
// for why an uncapped buffer is the exception, not the default, in this
// codebase.
inline constexpr std::size_t kUnboundedCapacity = static_cast<std::size_t>(-1);

// A contiguous, grow-capped byte buffer (spec decision D3 — see
// docs/decisions/0006-buffer-representation.md). Two properties, both load-
// bearing for how this codebase uses it:
//
//   - Contiguous: every unconsumed byte currently held is laid out in one
//     unbroken span (data() .. data()+size()), so a caller can hand a
//     single pointer+length straight to recv(2)/send(2) — no scatter/
//     gather, no per-fragment bookkeeping at the syscall boundary.
//   - Grow-capped: append() refuses to grow this buffer's *unconsumed* size
//     past max_capacity(), reporting that back via a bool return instead of
//     allocating without bound. This is what makes a buffer sitting on a
//     slow/adversarial peer's connection a bounded resource instead of an
//     unbounded one — see the spec's core design principle ("queues are
//     bounded") and agent execution rule 4 ("an unbounded queue is a
//     review-blocking defect"), both of which apply just as much to a
//     single accumulating buffer as to a queue of them.
//
// Not a ring buffer: append() always adds at the back; consume() always
// removes from the front. Internally backed by one std::vector<char> plus a
// read offset into it; consume() reclaims the consumed prefix's storage
// once keeping it around stops being worth an extra branch (see
// compact_if_worthwhile() in the .cpp) rather than on every call, so the
// common read-a-bit/consume-it-all/read-more cycle (this codebase's
// dominant access pattern — see NonblockingHttpServer::drive_parser and
// Connection::flush_output()) stays cheap: it degrades to `storage_.clear()`
// exactly when a buffer is fully drained, which is the overwhelmingly
// common case for both read_buffer and each OutputQueue chunk.
//
// Not thread-safe (single-reactor-thread ownership, like everything else in
// arcserve::reactor). Not copyable — buffers on this codebase's I/O path
// are large enough that an implicit copy is exactly the kind of allocation
// Phase 7's profiling work would want to catch, so this type doesn't allow
// one to happen by accident; movable.
class ByteBuffer {
 public:
  explicit ByteBuffer(std::size_t max_capacity = kUnboundedCapacity) noexcept
      : max_capacity_(max_capacity) {}

  ByteBuffer(const ByteBuffer&) = delete;
  ByteBuffer& operator=(const ByteBuffer&) = delete;
  ByteBuffer(ByteBuffer&&) noexcept = default;
  ByteBuffer& operator=(ByteBuffer&&) noexcept = default;
  ~ByteBuffer() = default;

  // Appends [data, data+length) to the back. All-or-nothing: if doing so
  // would make size() exceed max_capacity(), this buffer is left completely
  // unchanged and false is returned — never a partial append. Callers that
  // get false back must decide the overload policy themselves (this class
  // only enforces the bound; see reactor::OutputQueue and
  // docs/decisions/0006 for what the servers built on top of it do with a
  // false return).
  [[nodiscard]] bool append(const void* data, std::size_t length);
  [[nodiscard]] bool append(std::string_view data) { return append(data.data(), data.size()); }

  // Removes up to `n` bytes from the front (bytes the caller has already
  // sent, or already handed to a protocol parser). `n` is clamped to
  // size() — consuming more than is present is treated as "consume
  // everything" rather than undefined behavior, matching this codebase's
  // general preference (see FileDescriptor::reset()) for forgiving,
  // impossible-to-misuse-into-UB primitives at this layer.
  void consume(std::size_t n) noexcept;

  // Discards every held byte. Equivalent to consume(size()) but cheaper
  // (skips the compaction heuristic and always fully releases the backing
  // storage's logical length, though not necessarily its allocated
  // capacity — the underlying std::vector may still hold reusable
  // capacity, which is the point: the next append() need not reallocate).
  void clear() noexcept;

  [[nodiscard]] const char* data() const noexcept { return storage_.data() + read_offset_; }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size() - read_offset_; }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] std::string_view view() const noexcept { return {data(), size()}; }
  [[nodiscard]] std::size_t max_capacity() const noexcept { return max_capacity_; }

 private:
  void compact_if_worthwhile();

  std::vector<char> storage_;
  std::size_t read_offset_ = 0;
  std::size_t max_capacity_;
};

}  // namespace arcserve::buffers
