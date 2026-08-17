#include "arcserve/net/socket.hpp"

#include <fcntl.h>
#include <sys/socket.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <thread>

#include <gtest/gtest.h>

namespace arcserve::net {
namespace {

bool is_nonblocking(int fd) {
  int flags = ::fcntl(fd, F_GETFL);
  EXPECT_NE(flags, -1) << "fcntl(F_GETFL) failed: " << std::strerror(errno);
  return (flags & O_NONBLOCK) != 0;
}

// Repeatedly polls accept_nonblocking() for a short window: the connecting
// side may complete its handshake asynchronously relative to this call, so
// a single immediate attempt is not guaranteed to see it yet even on
// loopback.
std::optional<FileDescriptor> accept_nonblocking_with_retry(const TcpListener& listener) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    std::optional<FileDescriptor> accepted = listener.accept_nonblocking();
    if (accepted.has_value()) {
      return accepted;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return std::nullopt;
}

TEST(TcpListenerTest, BlockingByDefault) {
  TcpListener listener = TcpListener::listen(/*port=*/0);
  EXPECT_FALSE(is_nonblocking(listener.native_handle()));
}

TEST(TcpListenerTest, NonblockingParameterPutsListenerIntoNonblockingMode) {
  TcpListener listener = TcpListener::listen(/*port=*/0, /*backlog=*/128, /*nonblocking=*/true);
  EXPECT_TRUE(is_nonblocking(listener.native_handle()));
}

TEST(TcpListenerTest, NonblockingListenerStillResolvesEphemeralPort) {
  TcpListener listener = TcpListener::listen(0, 128, true);
  EXPECT_NE(listener.port(), 0);
}

TEST(TcpListenerTest, AcceptNonblockingReturnsNulloptWithNoPendingConnection) {
  TcpListener listener = TcpListener::listen(0, 128, /*nonblocking=*/true);
  EXPECT_FALSE(listener.accept_nonblocking().has_value());
}

TEST(TcpListenerTest, AcceptNonblockingReturnsAlreadyNonblockingClientFd) {
  TcpListener listener = TcpListener::listen(0, 128, /*nonblocking=*/true);
  FileDescriptor client_side = connect_loopback(listener.port());

  std::optional<FileDescriptor> accepted = accept_nonblocking_with_retry(listener);
  ASSERT_TRUE(accepted.has_value());
  EXPECT_TRUE(is_nonblocking(accepted->get()));
}

TEST(TcpListenerTest, AcceptNonblockingDrainsMultiplePendingConnections) {
  TcpListener listener = TcpListener::listen(0, 128, true);
  FileDescriptor client_a = connect_loopback(listener.port());
  FileDescriptor client_b = connect_loopback(listener.port());

  std::optional<FileDescriptor> accepted_a = accept_nonblocking_with_retry(listener);
  std::optional<FileDescriptor> accepted_b = accept_nonblocking_with_retry(listener);
  ASSERT_TRUE(accepted_a.has_value());
  ASSERT_TRUE(accepted_b.has_value());
  EXPECT_NE(accepted_a->get(), accepted_b->get());

  EXPECT_FALSE(listener.accept_nonblocking().has_value());
}

TEST(SetNonblockingTest, TurnsOnNonblockFlagWithoutClobberingOthers) {
  int fds[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  FileDescriptor a(fds[0]);
  FileDescriptor b(fds[1]);

  EXPECT_FALSE(is_nonblocking(a.get()));
  set_nonblocking(a.get());
  EXPECT_TRUE(is_nonblocking(a.get()));
  EXPECT_FALSE(is_nonblocking(b.get()));  // the other end is untouched
}

TEST(SetNonblockingTest, IsIdempotent) {
  int fds[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  FileDescriptor a(fds[0]);
  FileDescriptor b(fds[1]);

  set_nonblocking(a.get());
  set_nonblocking(a.get());  // must not throw or misbehave the second time
  EXPECT_TRUE(is_nonblocking(a.get()));
}

TEST(DescribePeerTest, ReturnsAddressPortForConnectedSocket) {
  TcpListener listener = TcpListener::listen(0, 128, true);
  FileDescriptor client = connect_loopback(listener.port());

  std::optional<FileDescriptor> accepted = accept_nonblocking_with_retry(listener);
  ASSERT_TRUE(accepted.has_value());

  std::string peer = describe_peer(accepted->get());
  EXPECT_NE(peer.find("127.0.0.1:"), std::string::npos) << "peer=" << peer;
}

TEST(DescribePeerTest, ReturnsUnknownForBadFd) { EXPECT_EQ(describe_peer(-1), "unknown"); }

TEST(ReadSomeTest, NonblockingSocketWithNoDataReturnsWouldBlockImmediately) {
  int fds[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  FileDescriptor a(fds[0]);
  FileDescriptor b(fds[1]);
  set_nonblocking(a.get());

  char buf[16];
  IoResult result = read_some(a.get(), buf, sizeof(buf));
  EXPECT_EQ(result.status, IoStatus::WouldBlock);
  EXPECT_EQ(result.bytes_transferred, 0u);
}

TEST(ReadSomeTest, NonblockingSocketReadsAvailableData) {
  int fds[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  FileDescriptor a(fds[0]);
  FileDescriptor b(fds[1]);
  set_nonblocking(a.get());

  ASSERT_EQ(::send(b.get(), "hi", 2, 0), 2);
  char buf[16] = {};
  IoResult result = read_some(a.get(), buf, sizeof(buf));
  EXPECT_EQ(result.status, IoStatus::Ok);
  EXPECT_EQ(result.bytes_transferred, 2u);
  EXPECT_EQ(std::string(buf, 2), "hi");
}

}  // namespace
}  // namespace arcserve::net
