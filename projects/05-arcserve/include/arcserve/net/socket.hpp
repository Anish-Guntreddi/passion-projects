#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "arcserve/net/file_descriptor.hpp"

namespace arcserve::net {

// Thrown for socket *setup* failures that cannot be handled locally: the
// syscalls involved in getting a listening or client socket into existence
// (socket/setsockopt/bind/listen/getsockname/connect). By the time a
// connection is established, ordinary read/write failures are reported via
// IoResult instead (see below) — those are expected, routine outcomes
// (peer closed, etc.), not exceptional setup failures.
class SocketError : public std::runtime_error {
 public:
  explicit SocketError(const std::string& what) : std::runtime_error(what) {}
};

// Outcome of a single read/write attempt against an already-connected
// socket. Partial transfers report `Ok` with bytes_transferred < the
// requested amount — per the spec's core design principle, partial
// reads/writes are normal, not errors, and callers are expected to loop.
enum class IoStatus {
  Ok,          // Some (possibly zero, for a zero-length request) bytes moved.
  Closed,      // Peer closed the connection: orderly EOF on read, or
               // EPIPE/ECONNRESET on write.
  WouldBlock,  // Nonblocking fd had nothing to read / no send-buffer space.
               // Never produced by the Phase 1 blocking server (its sockets
               // are never O_NONBLOCK); the enumerator exists so Phase 2's
               // nonblocking reactor can reuse this exact type.
  Error,       // Unrecoverable error; inspect errno immediately after the
               // call that produced this IoResult if diagnosing.
};

struct IoResult {
  std::size_t bytes_transferred = 0;
  IoStatus status = IoStatus::Ok;
};

// A listening TCP/IPv4 socket bound to `port` on all local interfaces
// (INADDR_ANY). Port 0 requests a kernel-assigned ephemeral port; call
// port() after construction to discover which one was picked (used by
// tests so they never hardcode a port number).
//
// Lifetime: owns its fd via FileDescriptor; the fd is closed exactly once,
// when the TcpListener is destroyed. Move-only, matching FileDescriptor.
class TcpListener {
 public:
  // `nonblocking`: if true, the listening socket itself is put into
  // O_NONBLOCK mode (via set_nonblocking, below) before this returns, and
  // accept_nonblocking() (not accept()) is the intended way to pull
  // connections off it — see that method's docs. Phase 1's BlockingHttpServer
  // never sets this (defaults to false, preserving its blocking accept()
  // loop unchanged); Phase 2's epoll reactor always does.
  [[nodiscard]] static TcpListener listen(std::uint16_t port, int backlog = 128,
                                           bool nonblocking = false);

  TcpListener(TcpListener&&) noexcept = default;
  TcpListener& operator=(TcpListener&&) noexcept = default;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;
  ~TcpListener() = default;

  // Blocking accept(2). Loops internally on EINTR so callers never observe
  // it. Throws SocketError on any other accept(2) failure. Only meaningful
  // when this listener is *not* in nonblocking mode (Phase 1 usage); a
  // nonblocking listener's accept() would still work syscall-wise but
  // defeats the purpose — use accept_nonblocking() instead.
  [[nodiscard]] FileDescriptor accept() const;

  // Nonblocking accept(2), for the epoll reactor's listener-readable path
  // (Phase 2). Only valid to call on a listener constructed with
  // `nonblocking = true`.
  //
  // Returns std::nullopt when there is nothing to accept right now — both
  // for the ordinary case (EAGAIN/EWOULDBLOCK: the backlog is empty) and
  // for transient resource-exhaustion conditions (EMFILE/ENFILE — this
  // process or the system is out of file descriptors — or a connection the
  // kernel already aborted before this call reached it: ECONNABORTED/
  // EPROTO). Treating those the same as "nothing pending yet" rather than
  // throwing is deliberate: any connection that caused EMFILE/ENFILE stays
  // in the kernel's listen backlog and is retried automatically on this
  // listener's next readable event (level-triggered epoll keeps reporting
  // it as long as the backlog is non-empty) — this self-throttles accept()
  // under fd exhaustion instead of crashing the reactor thread over a
  // condition the spec explicitly calls out as a failure case to handle,
  // not a fatal setup error (§1.6: "EMFILE/resource-exhaustion simulation
  // where feasible").
  //
  // Every other accept(2) failure (a real bug — e.g. EINVAL, EBADF) throws
  // SocketError, same as accept(). The returned client fd is always itself
  // already in O_NONBLOCK mode (accept(2) does not inherit that flag from
  // the listening socket) — callers never need to call set_nonblocking()
  // on it themselves.
  [[nodiscard]] std::optional<FileDescriptor> accept_nonblocking() const;

  // The fd underlying this listener, for use with poll()/select()/epoll.
  // Callers must not close it directly — ownership stays with this object.
  [[nodiscard]] int native_handle() const noexcept { return fd_.get(); }

  [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

 private:
  TcpListener(FileDescriptor fd, std::uint16_t port) noexcept : fd_(std::move(fd)), port_(port) {}

  FileDescriptor fd_;
  std::uint16_t port_ = 0;
};

// Opens a blocking TCP client socket connected to 127.0.0.1:port. This
// exists for tests (arcserve is a server, not a client — see
// docs/decisions/0002-protocol-scope-http-1.1-subset.md for scope). Throws
// SocketError on failure.
[[nodiscard]] FileDescriptor connect_loopback(std::uint16_t port);

// Puts `fd` into nonblocking mode (fcntl F_GETFL/F_SETFL with O_NONBLOCK
// added to whatever flags were already set — not a blind overwrite).
// Throws SocketError on failure. This plus the WouldBlock handling already
// built into read_some()/write_all() above is everything the epoll reactor
// (Phase 2) needs to make a socket reactor-safe.
void set_nonblocking(int fd);

// A human-readable "address:port" description of the peer connected on
// `fd` (getpeername(2) + inet_ntop), e.g. "127.0.0.1:54321". Returns
// "unknown" rather than throwing if getpeername(2) fails (this is
// diagnostic/observability metadata — FR2's "peer metadata" — not
// something a connection's correctness depends on, so a lookup failure
// must never be fatal to the connection). Never throws.
[[nodiscard]] std::string describe_peer(int fd) noexcept;

// Reads whatever is available (blocking) into buffer[0, capacity). Loops
// internally on EINTR. Never throws — all outcomes, including errors, are
// reported through the returned IoResult.
[[nodiscard]] IoResult read_some(int fd, void* buffer, std::size_t capacity) noexcept;

// Writes all of [data, data+length) to fd, looping over partial writes and
// EINTR until the whole buffer is sent or an error/close is observed.
// Never throws.
[[nodiscard]] IoResult write_all(int fd, const void* data, std::size_t length) noexcept;

}  // namespace arcserve::net
