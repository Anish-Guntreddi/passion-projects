#pragma once

#include <cstddef>
#include <cstdint>
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
  [[nodiscard]] static TcpListener listen(std::uint16_t port, int backlog = 128);

  TcpListener(TcpListener&&) noexcept = default;
  TcpListener& operator=(TcpListener&&) noexcept = default;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;
  ~TcpListener() = default;

  // Blocking accept(2). Loops internally on EINTR so callers never observe
  // it. Throws SocketError on any other accept(2) failure.
  [[nodiscard]] FileDescriptor accept() const;

  // The fd underlying this listener, for use with poll()/select(). Callers
  // must not close it directly — ownership stays with this object.
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

// Reads whatever is available (blocking) into buffer[0, capacity). Loops
// internally on EINTR. Never throws — all outcomes, including errors, are
// reported through the returned IoResult.
[[nodiscard]] IoResult read_some(int fd, void* buffer, std::size_t capacity) noexcept;

// Writes all of [data, data+length) to fd, looping over partial writes and
// EINTR until the whole buffer is sent or an error/close is observed.
// Never throws.
[[nodiscard]] IoResult write_all(int fd, const void* data, std::size_t length) noexcept;

}  // namespace arcserve::net
