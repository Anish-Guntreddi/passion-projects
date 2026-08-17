#pragma once

#include <sys/epoll.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "arcserve/net/file_descriptor.hpp"

namespace arcserve::reactor {

// Thrown for reactor *setup*/registration failures (epoll_create1,
// epoll_ctl, epoll_wait, eventfd) — mirrors net::SocketError's role for
// socket setup failures: these are unexpected, unrecoverable conditions,
// not routine per-connection outcomes.
class ReactorError : public std::runtime_error {
 public:
  explicit ReactorError(const std::string& what) : std::runtime_error(what) {}
};

// Interest/readiness bitmask. Deliberately just epoll's own EPOLLIN/EPOLLOUT
// values, not a reinvented portability enum — this project targets Linux's
// real epoll specifically (spec §1.8: "architecture explainable without
// Boost.Asio hiding the event model"), so there is nothing to gain by
// hiding which bits these are.
using InterestMask = std::uint32_t;
inline constexpr InterestMask kReadable = static_cast<InterestMask>(EPOLLIN);
inline constexpr InterestMask kWritable = static_cast<InterestMask>(EPOLLOUT);

// Invoked once per ready fd per epoll_wait() batch, with whatever subset of
// {kReadable, kWritable, EPOLLERR, EPOLLHUP} epoll reported (EPOLLERR/
// EPOLLHUP are always reported regardless of the registered interest mask
// — see man epoll_ctl(2) — so callers must check for them even if they
// only ever registered kReadable/kWritable).
using ReadyCallback = std::function<void(int fd, std::uint32_t ready_events)>;

// A thin, protocol-agnostic wrapper around Linux epoll, run level-triggered
// (spec decision D2 — see docs/decisions/0004-epoll-level-triggered.md).
// This class knows nothing about connections, buffers, or HTTP: it is
// exactly "fd + interest mask -> callback", plus a graceful-shutdown wakeup
// mechanism (FR7). EchoReactorServer (Phase 2) and NonblockingHttpServer
// (Phase 3) are both built on this same, unmodified class — see
// docs/decisions/0005-reactor-connection-model.md for why that separation
// is deliberate.
//
// Concurrency/lifetime invariants:
//   - add()/modify()/remove()/run() are NOT thread-safe against each other
//     and must only ever be called from the single thread that calls
//     run() — in every server built on this class, that means "from inside
//     a ReadyCallback", since accepting a new connection (add), rearming a
//     connection's interest (modify), and tearing one down (remove) all
//     happen as a direct consequence of a dispatched event. This is a
//     single-threaded reactor by design (spec: "one thread never blocks
//     the reactor"), not merely "usually single-threaded".
//   - stop() is the one exception: it is safe to call from any thread,
//     including concurrently with run() executing on another thread. It
//     sets an atomic flag run() checks between dispatch batches, and wakes
//     a run() currently blocked in epoll_wait(-1) via an eventfd write —
//     without this, a reactor with no other fd activity could otherwise
//     block in epoll_wait forever after stop() is requested.
//   - Every fd this reactor is told about via add() must stay open and
//     registered for as long as the caller wants events for it; ownership
//     of the fd itself is never taken by this class (callers — via
//     reactor::Connection/FileDescriptor — own their fds and are
//     responsible for calling remove() before/while closing them).
//   - Not copyable or movable: identity matters (the epoll fd, the wakeup
//     fd, and the callback map are all tied to this specific instance);
//     run() holding `this` implicitly for its whole duration is not safe
//     to invalidate via a move.
class EpollReactor {
 public:
  EpollReactor();
  ~EpollReactor() = default;

  EpollReactor(const EpollReactor&) = delete;
  EpollReactor& operator=(const EpollReactor&) = delete;
  EpollReactor(EpollReactor&&) = delete;
  EpollReactor& operator=(EpollReactor&&) = delete;

  // Registers `fd` for `interest` and stores `callback` to invoke on every
  // future ready event for it. Throws ReactorError if `fd` is already
  // registered (or any other epoll_ctl(2) failure). Reactor-thread only.
  void add(int fd, InterestMask interest, ReadyCallback callback);

  // Changes the interest mask for an already-registered `fd` (its callback
  // is unchanged). Throws ReactorError if `fd` is not registered (or any
  // other epoll_ctl(2) failure). Reactor-thread only.
  void modify(int fd, InterestMask interest);

  // Unregisters `fd`: no further events are dispatched for it, and its
  // callback is dropped. Safe to call on an fd that was never registered,
  // or already removed (a no-op in that case) — deliberately noexcept, so
  // every connection-teardown path in a server built on this class can
  // call it unconditionally without a try/catch. Reactor-thread only.
  void remove(int fd) noexcept;

  // Blocks the calling thread, dispatching ready events to their
  // registered callbacks, until stop() is observed. Waits with an
  // unbounded epoll_wait(-1) timeout between batches — this reactor relies
  // entirely on the eventfd wakeup for shutdown latency, not a bounded
  // poll interval (contrast BlockingHttpServer's Phase 1 approach, which
  // had no such wakeup available). Throws ReactorError on an unexpected
  // epoll_wait(2) failure (EINTR is retried transparently, not an error).
  void run();

  // Thread-safe (see class-level invariants above).
  void stop() noexcept;

  // Number of fds currently registered (excludes the internal wakeup fd).
  // Reactor-thread only (like add/modify/remove) — not synchronized for
  // cross-thread reads.
  [[nodiscard]] std::size_t registered_count() const noexcept { return callbacks_.size(); }

 private:
  void dispatch_ready(int fd, std::uint32_t events);

  net::FileDescriptor epoll_fd_;
  net::FileDescriptor wakeup_fd_;
  std::unordered_map<int, ReadyCallback> callbacks_;
  std::atomic<bool> stop_requested_{false};
};

}  // namespace arcserve::reactor
