#include "arcserve/net/socket.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>

namespace arcserve::net {

namespace {

[[noreturn]] void throw_errno(const char* syscall) {
  throw SocketError(std::string(syscall) + "(2) failed: " + std::strerror(errno));
}

}  // namespace

TcpListener TcpListener::listen(std::uint16_t port, int backlog) {
  FileDescriptor fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (!fd) {
    throw_errno("socket");
  }

  int reuse = 1;
  if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
    throw_errno("setsockopt(SO_REUSEADDR)");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  if (::bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    throw_errno("bind");
  }

  if (::listen(fd.get(), backlog) != 0) {
    throw_errno("listen");
  }

  // Re-query the bound address: when `port` was 0, the kernel just picked
  // one, and this is the only way to learn what it picked (tests rely on
  // this to avoid hardcoding a port number).
  socklen_t addr_len = sizeof(addr);
  if (::getsockname(fd.get(), reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0) {
    throw_errno("getsockname");
  }

  return TcpListener(std::move(fd), ntohs(addr.sin_port));
}

FileDescriptor TcpListener::accept() const {
  for (;;) {
    int client_fd = ::accept(fd_.get(), nullptr, nullptr);
    if (client_fd >= 0) {
      return FileDescriptor(client_fd);
    }
    if (errno == EINTR) {
      continue;
    }
    throw_errno("accept");
  }
}

FileDescriptor connect_loopback(std::uint16_t port) {
  FileDescriptor fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (!fd) {
    throw_errno("socket");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1) {
    throw SocketError("inet_pton(2) failed to parse 127.0.0.1");
  }

  for (;;) {
    if (::connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return fd;
    }
    if (errno == EINTR) {
      continue;
    }
    throw_errno("connect");
  }
}

IoResult read_some(int fd, void* buffer, std::size_t capacity) noexcept {
  for (;;) {
    ssize_t n = ::recv(fd, buffer, capacity, 0);
    if (n > 0) {
      return {static_cast<std::size_t>(n), IoStatus::Ok};
    }
    if (n == 0) {
      return {0, IoStatus::Closed};
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return {0, IoStatus::WouldBlock};
    }
    return {0, IoStatus::Error};
  }
}

IoResult write_all(int fd, const void* data, std::size_t length) noexcept {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t total_written = 0;

  while (total_written < length) {
    // MSG_NOSIGNAL: writing to a peer that has already reset the connection
    // reports failure via errno==EPIPE instead of raising SIGPIPE (whose
    // default disposition is process termination — not an acceptable
    // outcome for "a client disconnected").
    ssize_t n = ::send(fd, bytes + total_written, length - total_written, MSG_NOSIGNAL);
    if (n > 0) {
      total_written += static_cast<std::size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EPIPE || errno == ECONNRESET)) {
      return {total_written, IoStatus::Closed};
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return {total_written, IoStatus::WouldBlock};
    }
    return {total_written, IoStatus::Error};
  }
  return {total_written, IoStatus::Ok};
}

}  // namespace arcserve::net
