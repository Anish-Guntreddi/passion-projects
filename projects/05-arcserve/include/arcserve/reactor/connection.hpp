#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include "arcserve/buffers/byte_buffer.hpp"
#include "arcserve/net/file_descriptor.hpp"
#include "arcserve/reactor/output_queue.hpp"

namespace arcserve::reactor {

// A connection's coarse lifecycle state, driven entirely by which epoll
// interest is currently armed for its fd (see docs/decisions/0005):
//
//   kReading  only EPOLLIN armed. read_buffer accumulates bytes off the
//             socket; the protocol layer is waiting for more of them.
//   kWriting  the output queue has bytes queued that didn't fit in one
//             send(2) call; only EPOLLOUT is armed (reading is paused —
//             see the ADR for why) until it fully drains, at which point
//             the connection returns to kReading.
//   kClosing  identical to kWriting (draining the output queue under
//             EPOLLOUT) except the connection is torn down once fully
//             flushed instead of returning to kReading — used for the
//             final response before a "Connection: close", a keep-alive-
//             count limit, a parser error, or an output-queue overflow
//             (docs/decisions/0006-buffer-representation.md).
//   kClosed   terminal: already unregistered from the reactor. A
//             connection registry is expected to erase (and thereby close,
//             via ~Connection) an entry in this state promptly; nothing in
//             this codebase currently leaves one sitting in kClosed.
enum class ConnectionState { kReading, kWriting, kClosing, kClosed };

// Outcome of a single flush_output() call.
enum class FlushResult {
  kFlushed,  // The output queue is now empty; safe to return to kReading
             // (or, if a close was already pending, to tear the connection
             // down now).
  kPending,  // Some bytes remain queued (a send(2) call hit EAGAIN); caller
             // should arm EPOLLOUT and try again on the next writable
             // event.
  kFailed,   // The peer closed, reset the connection, or an unrecoverable
             // write error occurred; caller should tear the connection
             // down.
};

// Default cap on Connection::read_buffer: generous headroom over anything
// this codebase's own read loops (process_readable(), 8192-byte chunks
// looped until EAGAIN/a short read) accumulate before the protocol layer
// gets a chance to drain it, while still being a real, finite bound rather
// than buffers::kUnboundedCapacity — see docs/decisions/0006-buffer-
// representation.md.
inline constexpr std::size_t kDefaultMaxReadBufferBytes = 8 * 1024 * 1024;  // 8 MiB

// Generic per-connection state (FR2) shared by every reactor-driven server
// in this codebase: the accepted client fd (owned), accumulated unparsed
// input, a bounded multi-chunk output queue, a lifecycle state, peer
// metadata, and an activity timestamp for Phase 6's future idle-timeout
// sweep to consume. Deliberately protocol-agnostic — see
// docs/decisions/0005-reactor-connection-model.md for why this class knows
// nothing about HTTP (EchoReactorServer uses it directly; HttpConnection
// composes it with protocol::HttpRequestParser rather than this class
// growing a parser field of its own).
//
// Buffer representation (spec decision D3 — see
// docs/decisions/0006-buffer-representation.md): read_buffer is a
// contiguous, grow-capped buffers::ByteBuffer; output is a bounded,
// multi-chunk reactor::OutputQueue (each chunk itself a ByteBuffer). Both
// caps are configurable per-connection via the constructor, defaulting to
// generous values no existing test comes close to.
//
// Concurrency: not thread-safe, and not intended to be — a Connection is
// only ever touched from the single reactor thread that owns it (same
// invariant as EpollReactor itself). Every method on this class, including
// flush_output() (which performs the actual send(2) syscalls), must only
// ever be called from that one thread.
class Connection {
 public:
  // Member-initializer order below matches declaration order (read_buffer,
  // being public, is declared textually before the private fd_/peer_/
  // write_queue_ — see the fields below) rather than the more-natural-
  // reading fd/peer/read_buffer/write_queue order, purely to keep
  // -Wreorder satisfied; construction order has no observable effect here
  // since none of these members' constructors depend on another.
  Connection(net::FileDescriptor fd, std::string peer,
             std::size_t max_output_queue_bytes = kDefaultMaxOutputQueueBytes,
             std::size_t max_read_buffer_bytes = kDefaultMaxReadBufferBytes) noexcept
      : read_buffer(max_read_buffer_bytes),
        fd_(std::move(fd)),
        peer_(std::move(peer)),
        write_queue_(max_output_queue_bytes) {}

  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;
  // Not movable: every registry in this codebase stores Connections behind
  // std::unique_ptr/std::shared_ptr specifically so a Connection's address
  // never changes after construction. Reactor callbacks look a connection
  // up by fd (or, for Phase 5's worker-pool completions, a std::weak_ptr)
  // through that registry on every dispatch rather than closing over a raw
  // Connection*, so nothing today *requires* address stability — but
  // forbidding move removes an entire class of "relocated a Connection
  // while a callback held a reference to it" bugs a future caller could
  // otherwise introduce, at zero cost (nothing needs to move one).
  Connection(Connection&&) = delete;
  Connection& operator=(Connection&&) = delete;
  ~Connection() = default;

  [[nodiscard]] int fd() const noexcept { return fd_.get(); }
  [[nodiscard]] const std::string& peer() const noexcept { return peer_; }

  [[nodiscard]] ConnectionState state() const noexcept { return state_; }
  void set_state(ConnectionState state) noexcept { state_ = state; }

  [[nodiscard]] std::chrono::steady_clock::time_point last_activity() const noexcept {
    return last_activity_;
  }
  // Call whenever bytes are read from or written to this connection.
  // Nothing in Phases 2-4 acts on this timestamp yet; it exists now
  // because FR2 asks for it and Phase 6's idle-timeout sweep is its
  // intended future reader.
  void touch() noexcept { last_activity_ = std::chrono::steady_clock::now(); }

  // Convenience wrapper around output_queue().enqueue() — see OutputQueue's
  // docs for the bounded, all-or-nothing contract.
  [[nodiscard]] bool enqueue_output(std::string_view data) { return write_queue_.enqueue(data); }

  [[nodiscard]] OutputQueue& output_queue() noexcept { return write_queue_; }
  [[nodiscard]] const OutputQueue& output_queue() const noexcept { return write_queue_; }

  // Drains as much of the output queue as send(2) will currently accept,
  // looping across chunk boundaries (net::write_all() already loops within
  // one chunk, retrying past EINTR and partial sends until that chunk is
  // either fully sent or hits EAGAIN/an error) — see class docs for why
  // this may only be called from the reactor thread. Calling touch() on
  // any progress is this method's responsibility, not the caller's.
  [[nodiscard]] FlushResult flush_output() noexcept;

  // Unparsed bytes read off the socket, awaiting the protocol layer. Public
  // (like the original std::string field it replaces) — every reactor-
  // driven server in this codebase reads/mutates it directly as part of
  // its own read/parse loop; see docs/decisions/0006-buffer-representation.md.
  buffers::ByteBuffer read_buffer;

 private:
  net::FileDescriptor fd_;
  std::string peer_;
  ConnectionState state_ = ConnectionState::kReading;
  std::chrono::steady_clock::time_point last_activity_ = std::chrono::steady_clock::now();
  OutputQueue write_queue_;
};

}  // namespace arcserve::reactor
