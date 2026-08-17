#include "arcserve/reactor/epoll_reactor.hpp"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>

namespace arcserve::reactor {

namespace {

[[noreturn]] void throw_errno(const char* syscall) {
  throw ReactorError(std::string(syscall) + "(2) failed: " + std::strerror(errno));
}

}  // namespace

EpollReactor::EpollReactor()
    : epoll_fd_(::epoll_create1(EPOLL_CLOEXEC)),
      wakeup_fd_(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
  if (!epoll_fd_) {
    throw_errno("epoll_create1");
  }
  if (!wakeup_fd_) {
    throw_errno("eventfd");
  }

  epoll_event event{};
  event.events = EPOLLIN;
  event.data.fd = wakeup_fd_.get();
  if (::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_ADD, wakeup_fd_.get(), &event) != 0) {
    throw_errno("epoll_ctl(ADD wakeup_fd)");
  }
}

void EpollReactor::add(int fd, InterestMask interest, ReadyCallback callback) {
  epoll_event event{};
  event.events = interest;
  event.data.fd = fd;
  if (::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_ADD, fd, &event) != 0) {
    throw_errno("epoll_ctl(ADD)");
  }
  callbacks_[fd] = std::move(callback);
}

void EpollReactor::modify(int fd, InterestMask interest) {
  epoll_event event{};
  event.events = interest;
  event.data.fd = fd;
  if (::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_MOD, fd, &event) != 0) {
    throw_errno("epoll_ctl(MOD)");
  }
}

void EpollReactor::remove(int fd) noexcept {
  // Ignore the epoll_ctl(2) result: this is a best-effort unregistration
  // called from connection-teardown paths (including "the fd may already
  // be invalid/closed" paths), and it must never throw out of a noexcept
  // function. Closing a fd already implicitly removes it from any epoll
  // instance, so a failing DEL here is expected, not a bug.
  ::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_DEL, fd, nullptr);
  callbacks_.erase(fd);
}

void EpollReactor::stop() noexcept {
  stop_requested_.store(true, std::memory_order_release);
  std::uint64_t one = 1;
  // Best-effort: EpollReactor::stop() is noexcept, so there is nothing
  // actionable to do here if the write itself somehow fails. run()'s
  // stop_requested_ check still eventually fires on the reactor's next
  // otherwise-scheduled wakeup; this would only be a latency regression,
  // never a correctness one.
  (void)::write(wakeup_fd_.get(), &one, sizeof(one));
}

void EpollReactor::run() {
  // Deliberately NOT reset here. A stop() on another thread can land
  // between a caller deciding to spawn the reactor thread and this
  // function's first instruction actually running (every server built on
  // this class does exactly that in its constructor/destructor pair) — if
  // run() unconditionally cleared stop_requested_ at entry, that race would
  // silently discard an already-pending stop() and then park in
  // epoll_wait(-1) forever, even though the eventfd wakeup byte from that
  // earlier stop() was already drained on the first, immediate wakeup. The
  // flag is reset instead at the very end of this function (below), right
  // before returning, so the *next* run() call still starts clean — see
  // RunCanBeRestartedAfterStop.
  std::array<epoll_event, 64> events{};

  while (!stop_requested_.load(std::memory_order_acquire)) {
    int ready = ::epoll_wait(epoll_fd_.get(), events.data(), static_cast<int>(events.size()), -1);
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw_errno("epoll_wait");
    }

    for (int i = 0; i < ready; ++i) {
      const epoll_event& ready_event = events[static_cast<std::size_t>(i)];
      int fd = ready_event.data.fd;

      if (fd == wakeup_fd_.get()) {
        std::uint64_t drain = 0;
        while (::read(wakeup_fd_.get(), &drain, sizeof(drain)) > 0) {
          // Drain every pending wakeup tick; eventfd coalesces repeated
          // writes into a single counter, so in practice this loop runs
          // at most twice, but looping to EAGAIN is the only correct way
          // to fully drain it.
        }
        continue;
      }

      dispatch_ready(fd, ready_event.events);
    }
  }

  stop_requested_.store(false, std::memory_order_release);
}

void EpollReactor::dispatch_ready(int fd, std::uint32_t events) {
  auto it = callbacks_.find(fd);
  if (it == callbacks_.end()) {
    return;  // Defensive: nothing in this codebase currently removes a
             // *different* fd's callback mid-batch, but dispatch makes no
             // assumption that it never will.
  }
  // Copy the callback out before invoking it: the callback itself is free
  // to call add()/modify()/remove() on this reactor (a connection handler
  // closing its own connection is the common case), any of which can
  // rehash or erase from callbacks_ — invoking through a reference/
  // iterator into that map while it does so would be a use-after-free.
  ReadyCallback callback = it->second;
  callback(fd, events);
}

}  // namespace arcserve::reactor
