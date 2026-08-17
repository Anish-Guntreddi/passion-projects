#include "arcserve/net/file_descriptor.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include <gtest/gtest.h>

namespace arcserve::net {
namespace {

// True iff `fd` is not (or no longer) an open descriptor in this process.
bool is_closed(int fd) { return ::fcntl(fd, F_GETFD) == -1 && errno == EBADF; }

// Returns a pair of connected pipe fds the caller owns as raw ints (tests
// deliberately don't wrap both ends in FileDescriptor, so each test can
// hand exactly one end to the class under test and manage the other
// directly).
std::pair<int, int> make_pipe() {
  int fds[2];
  EXPECT_EQ(::pipe(fds), 0) << "pipe(2) failed: " << std::strerror(errno);
  return {fds[0], fds[1]};
}

TEST(FileDescriptorTest, DefaultConstructedIsInvalid) {
  FileDescriptor fd;
  EXPECT_FALSE(fd.valid());
  EXPECT_FALSE(static_cast<bool>(fd));
  EXPECT_EQ(fd.get(), -1);
}

TEST(FileDescriptorTest, TakesOwnershipAndReportsValid) {
  auto [read_end, write_end] = make_pipe();
  {
    FileDescriptor fd(read_end);
    EXPECT_TRUE(fd.valid());
    EXPECT_TRUE(static_cast<bool>(fd));
    EXPECT_EQ(fd.get(), read_end);
  }
  // Destructor must have closed it exactly once.
  EXPECT_TRUE(is_closed(read_end));
  ::close(write_end);
}

TEST(FileDescriptorTest, DestructorClosesOwnedFd) {
  auto [read_end, write_end] = make_pipe();
  { FileDescriptor fd(read_end); }
  EXPECT_TRUE(is_closed(read_end));
  ::close(write_end);
}

TEST(FileDescriptorTest, MoveConstructorTransfersOwnership) {
  auto [read_end, write_end] = make_pipe();
  FileDescriptor original(read_end);

  FileDescriptor moved(std::move(original));
  EXPECT_FALSE(original.valid());   // moved-from: owns nothing
  EXPECT_EQ(original.get(), -1);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.get(), read_end);

  // Only one destructor call should close read_end; if ownership were
  // duplicated instead of transferred, this would double-close.
  moved.reset();
  EXPECT_TRUE(is_closed(read_end));
  ::close(write_end);
}

TEST(FileDescriptorTest, MoveAssignmentClosesPreviousAndTakesNew) {
  auto [read_a, write_a] = make_pipe();
  auto [read_b, write_b] = make_pipe();

  FileDescriptor fd_a(read_a);
  FileDescriptor fd_b(read_b);

  fd_a = std::move(fd_b);

  EXPECT_TRUE(is_closed(read_a));  // fd_a's original fd was closed by the assignment
  EXPECT_TRUE(fd_a.valid());
  EXPECT_EQ(fd_a.get(), read_b);
  EXPECT_FALSE(fd_b.valid());  // moved-from

  ::close(write_a);
  ::close(write_b);
}

TEST(FileDescriptorTest, SelfMoveAssignmentLeavesFdOwnedAndOpen) {
  auto [read_end, write_end] = make_pipe();
  FileDescriptor fd(read_end);

  FileDescriptor* alias = &fd;  // avoid a syntactically-literal `fd = std::move(fd)`
  fd = std::move(*alias);

  EXPECT_TRUE(fd.valid());
  EXPECT_EQ(fd.get(), read_end);
  EXPECT_FALSE(is_closed(read_end));

  ::close(write_end);
}

TEST(FileDescriptorTest, ReleaseGivesUpOwnershipWithoutClosing) {
  auto [read_end, write_end] = make_pipe();
  FileDescriptor fd(read_end);

  int released = fd.release();
  EXPECT_EQ(released, read_end);
  EXPECT_FALSE(fd.valid());
  EXPECT_FALSE(is_closed(read_end));  // still open: release() must not close it

  ::close(released);
  ::close(write_end);
}

TEST(FileDescriptorTest, ReleaseOnEmptyReturnsNegativeOne) {
  FileDescriptor fd;
  EXPECT_EQ(fd.release(), -1);
}

TEST(FileDescriptorTest, ResetClosesCurrentAndTakesNew) {
  auto [read_a, write_a] = make_pipe();
  auto [read_b, write_b] = make_pipe();

  FileDescriptor fd(read_a);
  int close_result = fd.reset(read_b);

  EXPECT_EQ(close_result, 0);
  EXPECT_TRUE(is_closed(read_a));
  EXPECT_EQ(fd.get(), read_b);

  ::close(write_a);
  ::close(write_b);
}

TEST(FileDescriptorTest, ResetWithNoArgumentJustClosesAndInvalidates) {
  auto [read_end, write_end] = make_pipe();
  FileDescriptor fd(read_end);

  fd.reset();
  EXPECT_FALSE(fd.valid());
  EXPECT_TRUE(is_closed(read_end));
  ::close(write_end);
}

TEST(FileDescriptorTest, NegativeSentinelIsNeverPassedToClose) {
  // -1 must never reach ::close(); reset() on an empty FileDescriptor
  // reports success (0) rather than the -1/EBADF that ::close(-1) would
  // produce, proving close() was not called at all.
  FileDescriptor fd(-1);
  EXPECT_FALSE(fd.valid());
  EXPECT_EQ(fd.reset(), 0);
}

}  // namespace
}  // namespace arcserve::net
