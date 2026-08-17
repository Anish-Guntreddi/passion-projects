#include "arcserve/reactor/epoll_reactor.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

namespace arcserve::reactor {
namespace {

std::pair<int, int> make_pipe() {
  int fds[2];
  EXPECT_EQ(::pipe(fds), 0);
  return {fds[0], fds[1]};
}

TEST(EpollReactorTest, DispatchesReadableEventForPipe) {
  auto [read_end, write_end] = make_pipe();
  EpollReactor reactor;

  bool fired = false;
  std::uint32_t seen_events = 0;
  reactor.add(read_end, kReadable, [&](int fd, std::uint32_t events) {
    EXPECT_EQ(fd, read_end);
    fired = true;
    seen_events = events;
    reactor.stop();
  });

  ASSERT_EQ(::write(write_end, "x", 1), 1);
  reactor.run();

  EXPECT_TRUE(fired);
  EXPECT_TRUE((seen_events & kReadable) != 0);
  ::close(read_end);
  ::close(write_end);
}

// Level-triggered semantics (spec decision D2; docs/decisions/0004): if a
// callback doesn't drain a readable fd, the *next* run() iteration must be
// notified again immediately, with no explicit re-arm — the defining
// difference from edge-triggered mode.
TEST(EpollReactorTest, LevelTriggeredFiresAgainIfNotDrained) {
  auto [read_end, write_end] = make_pipe();
  EpollReactor reactor;

  int fire_count = 0;
  reactor.add(read_end, kReadable, [&](int /*fd*/, std::uint32_t /*events*/) {
    ++fire_count;
    if (fire_count >= 2) {
      reactor.stop();
      return;
    }
    // Deliberately do NOT read the pending byte.
  });

  ASSERT_EQ(::write(write_end, "x", 1), 1);
  reactor.run();

  EXPECT_EQ(fire_count, 2);
  ::close(read_end);
  ::close(write_end);
}

TEST(EpollReactorTest, RemoveStopsFurtherDispatchForThatFd) {
  auto [read_end, write_end] = make_pipe();
  EpollReactor reactor;

  int fire_count = 0;
  reactor.add(read_end, kReadable, [&](int fd, std::uint32_t /*events*/) {
    ++fire_count;
    char byte = 0;
    ASSERT_EQ(::read(fd, &byte, 1), 1);
    reactor.remove(fd);
    reactor.stop();
  });

  ASSERT_EQ(::write(write_end, "xy", 2), 2);
  reactor.run();

  EXPECT_EQ(fire_count, 1);
  ::close(read_end);
  ::close(write_end);
}

// Proves modify() actually changes what epoll reports, not merely that it
// doesn't crash: after dropping interest to 0, a second byte written to
// the same fd must not trigger a second dispatch, even though the reactor
// keeps running (driven by a timed stop() on another thread) long enough
// that a bug here would have plenty of opportunity to fire.
TEST(EpollReactorTest, ModifyToEmptyInterestSuppressesFurtherDispatch) {
  auto [read_end, write_end] = make_pipe();
  EpollReactor reactor;

  int fire_count = 0;
  reactor.add(read_end, kReadable, [&](int fd, std::uint32_t /*events*/) {
    ++fire_count;
    char byte = 0;
    ASSERT_EQ(::read(fd, &byte, 1), 1);
    if (fire_count == 1) {
      reactor.modify(fd, 0);
    }
  });

  ASSERT_EQ(::write(write_end, "x", 1), 1);

  std::thread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    reactor.stop();
  });
  ASSERT_EQ(::write(write_end, "y", 1), 1);
  reactor.run();
  stopper.join();

  EXPECT_EQ(fire_count, 1);
  ::close(read_end);
  ::close(write_end);
}

TEST(EpollReactorTest, RegisteredCountTracksAddAndRemove) {
  auto [read_end, write_end] = make_pipe();
  EpollReactor reactor;
  EXPECT_EQ(reactor.registered_count(), 0u);

  reactor.add(read_end, kReadable, [](int, std::uint32_t) {});
  EXPECT_EQ(reactor.registered_count(), 1u);

  reactor.remove(read_end);
  EXPECT_EQ(reactor.registered_count(), 0u);

  ::close(read_end);
  ::close(write_end);
}

TEST(EpollReactorTest, RemoveOnUnregisteredFdIsANoOp) {
  EpollReactor reactor;
  reactor.remove(12345);  // never added; must not throw
  EXPECT_EQ(reactor.registered_count(), 0u);
}

// FR7 (graceful shutdown / reactor wakeup mechanism), tested at the
// concurrency boundary directly relevant to it: stop() called from a
// different thread than run(). TSan-relevant per spec §1.6 ("worker
// shutdown ... TSan-compatible tests") — the same discipline applies to
// the reactor's own shutdown coordination.
TEST(EpollReactorTest, StopFromAnotherThreadWakesBlockedRunPromptly) {
  EpollReactor reactor;  // no fds registered: run() parks purely on the
                          // internal wakeup fd until stop() is called.

  std::atomic<bool> run_returned{false};
  std::thread runner([&] {
    reactor.run();
    run_returned.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let run() reach epoll_wait
  auto start = std::chrono::steady_clock::now();
  reactor.stop();
  runner.join();
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_TRUE(run_returned.load(std::memory_order_acquire));
  EXPECT_LT(elapsed, std::chrono::milliseconds(1000));
}

// Regression: stop() called before run() ever starts — a realistic
// scheduling race, not a contrived one. Every server built on this class
// spawns the reactor thread in its constructor and calls stop() in its
// destructor; if that destructor is ever reached before the spawned thread
// has actually entered run() (fast test teardown, an exception right after
// construction, etc.), stop() lands first. The buggy version of run()
// unconditionally reset stop_requested_ to false at entry, silently
// clobbering this already-pending stop() and then blocking in
// epoll_wait(-1) forever — even though the eventfd wakeup byte from the
// earlier stop() had already been drained on the first, immediate wakeup.
// This test has no timeout of its own; it relies on the unit-test binary's
// ctest-level TIMEOUT (see tests/CMakeLists.txt) to fail loudly rather than
// hang forever if this regresses.
TEST(EpollReactorTest, StopRequestedBeforeRunStillReturnsPromptly) {
  EpollReactor reactor;
  reactor.stop();  // no thread has called run() yet

  std::atomic<bool> run_returned{false};
  std::thread runner([&] {
    reactor.run();
    run_returned.store(true, std::memory_order_release);
  });
  runner.join();

  EXPECT_TRUE(run_returned.load(std::memory_order_acquire));
}

// A second stop()+run() cycle must behave the same as the first — the
// reactor is reusable, not single-shot.
TEST(EpollReactorTest, RunCanBeRestartedAfterStop) {
  EpollReactor reactor;
  int total_fired = 0;

  auto [read_end, write_end] = make_pipe();
  reactor.add(read_end, kReadable, [&](int fd, std::uint32_t) {
    char byte = 0;
    ASSERT_EQ(::read(fd, &byte, 1), 1);
    ++total_fired;
    reactor.stop();
  });

  ASSERT_EQ(::write(write_end, "a", 1), 1);
  reactor.run();
  EXPECT_EQ(total_fired, 1);

  ASSERT_EQ(::write(write_end, "b", 1), 1);
  reactor.run();
  EXPECT_EQ(total_fired, 2);

  ::close(read_end);
  ::close(write_end);
}

}  // namespace
}  // namespace arcserve::reactor
