#include "arcserve/reactor/connection.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <string>
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

bool is_closed(int fd) { return ::fcntl(fd, F_GETFD) == -1; }

TEST(ConnectionTest, ConstructedWithReadingStateAndEmptyBuffers) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "127.0.0.1:1234");

  EXPECT_EQ(conn.fd(), read_end);
  EXPECT_EQ(conn.peer(), "127.0.0.1:1234");
  EXPECT_EQ(conn.state(), ConnectionState::kReading);
  EXPECT_TRUE(conn.read_buffer.empty());
  EXPECT_TRUE(conn.write_buffer.empty());

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

TEST(ConnectionTest, ReadAndWriteBuffersAreIndependentlyMutable) {
  auto [read_end, write_end] = make_pipe();
  Connection conn(net::FileDescriptor(read_end), "peer");

  conn.read_buffer = "partial request";
  conn.write_buffer = "pending response";
  EXPECT_EQ(conn.read_buffer, "partial request");
  EXPECT_EQ(conn.write_buffer, "pending response");

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
