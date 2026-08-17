#pragma once

#include <chrono>
#include <string>
#include <utility>

#include "arcserve/net/file_descriptor.hpp"

namespace arcserve::reactor {

// A connection's coarse lifecycle state, driven entirely by which epoll
// interest is currently armed for its fd (see docs/decisions/0005):
//
//   kReading  only EPOLLIN armed. read_buffer accumulates bytes off the
//             socket; the protocol layer is waiting for more of them.
//   kWriting  write_buffer has bytes queued that didn't fit in one send(2)
//             call; only EPOLLOUT is armed (reading is paused — see the
//             ADR for why) until it fully drains, at which point the
//             connection returns to kReading.
//   kClosing  identical to kWriting (draining write_buffer under EPOLLOUT)
//             except the connection is torn down once fully flushed
//             instead of returning to kReading — used for the final
//             response before a "Connection: close", a keep-alive-count
//             limit, or a parser error.
//   kClosed   terminal: already unregistered from the reactor. A
//             connection registry is expected to erase (and thereby close,
//             via ~Connection) an entry in this state promptly; nothing in
//             this codebase currently leaves one sitting in kClosed.
enum class ConnectionState { kReading, kWriting, kClosing, kClosed };

// Generic per-connection state (FR2) shared by every reactor-driven server
// in this codebase: the accepted client fd (owned), accumulated unparsed
// input, buffered-but-not-yet-flushed output, a lifecycle state, peer
// metadata, and an activity timestamp for Phase 6's future idle-timeout
// sweep to consume. Deliberately protocol-agnostic — see
// docs/decisions/0005-reactor-connection-model.md for why this class knows
// nothing about HTTP (EchoReactorServer uses it directly; HttpConnection
// composes it with protocol::HttpRequestParser rather than this class
// growing a parser field of its own).
//
// Concurrency: not thread-safe, and not intended to be — a Connection is
// only ever touched from the single reactor thread that owns it (same
// invariant as EpollReactor itself).
class Connection {
 public:
  Connection(net::FileDescriptor fd, std::string peer) noexcept
      : fd_(std::move(fd)), peer_(std::move(peer)) {}

  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;
  // Not movable: every registry in this codebase stores Connections behind
  // std::unique_ptr specifically so a Connection's address never changes
  // after construction. Reactor callbacks look a connection up by fd
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
  // Nothing in Phases 2-3 acts on this timestamp yet; it exists now
  // because FR2 asks for it and Phase 6's idle-timeout sweep is its
  // intended future reader.
  void touch() noexcept { last_activity_ = std::chrono::steady_clock::now(); }

  // Unparsed bytes read off the socket, awaiting the protocol layer.
  std::string read_buffer;
  // Serialized bytes queued to send; index 0 is the next byte due on the
  // wire. A single growable buffer, not a queue of discrete messages or a
  // high-water-mark-bounded structure — see
  // docs/decisions/0005-reactor-connection-model.md for the reasoning and
  // for what Phase 4 is expected to formalize on top of this field.
  std::string write_buffer;

 private:
  net::FileDescriptor fd_;
  std::string peer_;
  ConnectionState state_ = ConnectionState::kReading;
  std::chrono::steady_clock::time_point last_activity_ = std::chrono::steady_clock::now();
};

}  // namespace arcserve::reactor
