#include "arcserve/reactor/connection.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "arcserve/net/file_descriptor.hpp"

namespace arcserve::reactor {
namespace {

std::pair<int, int> make_pipe() {
  int fds[2];
  EXPECT_EQ(::pipe(fds), 0);
  return {fds[0], fds[1]};
}

// A connected pair of AF_UNIX SOCK_STREAM fds: unlike a pipe, both ends
// support send(2)/recv(2) (what net::write_all()/read_some() actually
// call), which is what flush_output() tests below need — a Connection's fd
// must genuinely be a socket, not just any readable/writable fd.
std::pair<net::FileDescriptor, net::FileDescriptor> make_socketpair() {
  int fds[2];
  EXPECT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  return {net::FileDescriptor(fds[0]), net::FileDescriptor(fds[1])};
}

bool is_closed(int fd) { return ::fcntl(fd, F_GETFD) == -1; }

TEST(ConnectionTest, ConstructedWithReadingStateAndEmptyBuffers) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "127.0.0.1:1234");

  EXPECT_EQ(conn.fd(), read_end);
  EXPECT_EQ(conn.peer(), "127.0.0.1:1234");
  EXPECT_EQ(conn.state(), ConnectionState::kReading);
  EXPECT_TRUE(conn.read_buffer.empty());
  EXPECT_TRUE(conn.output_queue().empty());

  ::close(write_end);
}

TEST(ConnectionTest, SetStateChangesObservableState) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer");

  conn.set_state(ConnectionState::kWriting);
  EXPECT_EQ(conn.state(), ConnectionState::kWriting);
  conn.set_state(ConnectionState::kClosing);
  EXPECT_EQ(conn.state(), ConnectionState::kClosing);
  conn.set_state(ConnectionState::kClosed);
  EXPECT_EQ(conn.state(), ConnectionState::kClosed);

  ::close(write_end);
}

TEST(ConnectionTest, ReadBufferAndOutputQueueAreIndependentlyMutable) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer");

  ASSERT_TRUE(conn.read_buffer.append(std::string_view("partial request")));
  ASSERT_TRUE(conn.enqueue_output("pending response"));
  EXPECT_EQ(conn.read_buffer.view(), "partial request");
  EXPECT_EQ(conn.output_queue().queued_bytes(), std::string_view("pending response").size());

  ::close(write_end);
}

TEST(ConnectionTest, FlushOutputWithEmptyQueueReturnsFlushed) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer");

  EXPECT_EQ(conn.flush_output(), FlushResult::kFlushed);

  ::close(write_end);
}

TEST(ConnectionTest, FlushOutputSendsAllQueuedChunksInOrder) {
  auto [local, remote] = make_socketpair();
  Connection conn(std::move(local), "peer");

  ASSERT_TRUE(conn.enqueue_output("abc"));
  ASSERT_TRUE(conn.enqueue_output("def"));
  EXPECT_EQ(conn.output_queue().queued_chunks(), 2u);

  EXPECT_EQ(conn.flush_output(), FlushResult::kFlushed);
  EXPECT_TRUE(conn.output_queue().empty());

  char buf[16] = {};
  ssize_t n = ::recv(remote.get(), buf, sizeof(buf), 0);
  ASSERT_EQ(n, 6);
  EXPECT_EQ(std::string_view(buf, static_cast<std::size_t>(n)), "abcdef");
}

TEST(ConnectionTest, FlushOutputReportsFailedOncePeerCloses) {
  auto [local, remote] = make_socketpair();
  Connection conn(std::move(local), "peer");
  remote.reset();  // close the peer end before anything is ever sent

  ASSERT_TRUE(conn.enqueue_output("hello"));
  EXPECT_EQ(conn.flush_output(), FlushResult::kFailed);
}

TEST(ConnectionTest, EnqueueOutputRejectedWhenOverCapacityLeavesQueueUnchanged) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer", /*max_output_queue_bytes=*/8);

  EXPECT_TRUE(conn.enqueue_output("small"));  // 5 bytes, within the 8-byte cap
  EXPECT_FALSE(conn.enqueue_output("this is far too much data"));
  EXPECT_EQ(conn.output_queue().queued_bytes(), 5u);
  EXPECT_EQ(conn.output_queue().queued_chunks(), 1u);

  ::close(write_end);
}

TEST(ConnectionTest, ReadBufferAppendRejectedWhenOverCapacity) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer", reactor::kDefaultMaxOutputQueueBytes,
                   /*max_read_buffer_bytes=*/4);

  EXPECT_TRUE(conn.read_buffer.append(std::string_view("abcd")));
  EXPECT_FALSE(conn.read_buffer.append(std::string_view("e")));
  EXPECT_EQ(conn.read_buffer.view(), "abcd");

  ::close(write_end);
}

TEST(ConnectionTest, TouchAdvancesLastActivity) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer");
  auto initial = conn.last_activity();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  conn.touch();
  EXPECT_GT(conn.last_activity(), initial);

  ::close(write_end);
}

TEST(ConnectionTest, DestructorClosesOwnedFd) {
  auto [read_end, write_end] = make_pipe();
  { Connection conn(net::FileDescriptor(read_end), "peer"); }
  EXPECT_TRUE(is_closed(read_end));
  ::close(write_end);
}

}  // namespace
}  // namespace arcserve::reactor
