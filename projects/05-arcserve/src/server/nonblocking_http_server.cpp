#include "arcserve/server/nonblocking_http_server.hpp"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arcserve::server {

NonblockingHttpServer::NonblockingHttpServer(Config config, RequestHandler handler)
    : listener_(net::TcpListener::listen(config.port, config.backlog, /*nonblocking=*/true)),
      handler_(std::move(handler)),
      config_(config) {
  reactor_.add(listener_.native_handle(), reactor::kReadable,
               [this](int fd, std::uint32_t events) { on_listener_readable(fd, events); });

  if (config_.worker_pool != nullptr) {
    // See EpollReactor's own constructor for the identical eventfd-as-
    // cross-thread-wakeup pattern this mirrors; the difference here is
    // *what* wakes the reactor up (a completed worker result, not a
    // stop() request) and what runs once it does (draining
    // completion_queue_, not just observing a flag).
    completion_fd_ = net::FileDescriptor(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC));
    if (!completion_fd_) {
      throw reactor::ReactorError(std::string("eventfd(2) failed: ") + std::strerror(errno));
    }
    completion_queue_.emplace(config_.completion_queue_capacity);
    reactor_.add(completion_fd_.get(), reactor::kReadable,
                 [this](int fd, std::uint32_t events) { on_completion_readable(fd, events); });
  }
}

void NonblockingHttpServer::run() {
  reactor_.run();
  if (completion_queue_.has_value()) {
    // Closing here — right after the reactor loop has genuinely exited, on
    // the reactor thread itself — matters now that a worker thread can
    // block inside completion_queue_->push() (see dispatch_to_worker()'s
    // worker lambda): once run() has returned, on_completion_readable()
    // will never run again, so any worker currently (or later) blocked
    // waiting for push() to find space must be woken (push() returns false
    // on a closed queue instead of waiting forever) or it — and the
    // WorkerPool::stop() join the documented shutdown ordering requires the
    // caller to perform right after this call returns — would hang. Safe to
    // drop whatever's still queued or still arriving: by the class docs'
    // required shutdown ordering, no caller may still be waiting on a live
    // connection once run() has returned.
    completion_queue_->close();
  }
}

void NonblockingHttpServer::stop() noexcept { reactor_.stop(); }

void NonblockingHttpServer::on_listener_readable(int /*fd*/, std::uint32_t /*events*/) {
  for (;;) {
    std::optional<net::FileDescriptor> client = listener_.accept_nonblocking();
    if (!client.has_value()) {
      return;
    }

    int client_fd = client->get();
    std::string peer = net::describe_peer(client_fd);
    // shared_ptr, not unique_ptr: Phase 5's worker-dispatch mode needs a
    // std::weak_ptr to safely hand a request off across threads without
    // risking delivering a stale response to an fd that got reused after
    // this connection closed — see this class's "Worker-pool dispatch
    // mode" docs.
    auto connection = std::make_shared<HttpConnection>(std::move(*client), std::move(peer),
                                                         config_.max_output_queue_bytes,
                                                         config_.max_read_buffer_bytes);
    connections_.emplace(client_fd, std::move(connection));
    connections_accepted_.fetch_add(1, std::memory_order_relaxed);

    reactor_.add(client_fd, reactor::kReadable,
                 [this](int fd, std::uint32_t events) { on_client_event(fd, events); });
  }
}

void NonblockingHttpServer::on_client_event(int fd, std::uint32_t events) {
  auto it = connections_.find(fd);
  if (it == connections_.end()) {
    return;  // Defensive: see EpollReactor::dispatch_ready's comment.
  }
  // A local copy (not a reference to it->second) so this connection stays
  // alive for the rest of this call even if some path below erases the
  // registry entry partway through — matters more than it used to now
  // that this function may lead into dispatch_to_worker(), which needs a
  // shared_ptr to construct a weak_ptr from.
  std::shared_ptr<HttpConnection> conn_ptr = it->second;
  HttpConnection& conn = *conn_ptr;

  if ((events & (EPOLLERR | EPOLLHUP)) != 0) {
    close_connection(fd);
    return;
  }

  if ((events & reactor::kWritable) != 0) {
    reactor::FlushResult result = conn.base.flush_output();
    if (result == reactor::FlushResult::kFailed) {
      close_connection(fd);
      return;
    }
    if (result == reactor::FlushResult::kFlushed) {
      if (conn.base.state() == reactor::ConnectionState::kClosing) {
        close_connection(fd);
        return;
      }
      conn.base.set_state(reactor::ConnectionState::kReading);
      reactor_.modify(fd, reactor::kReadable);
    }
    // kPending: stay in kWriting/kClosing with EPOLLOUT still armed.
  }

  // Interest is never kReadable at the same time as kWriting/kClosing (see
  // reactor::ConnectionState's docs), so this state guard is really just
  // making that invariant load-bearing rather than implicit.
  if ((events & reactor::kReadable) != 0 &&
      conn.base.state() == reactor::ConnectionState::kReading) {
    process_readable(conn_ptr);
  }
}

void NonblockingHttpServer::on_completion_readable(int fd, std::uint32_t /*events*/) {
  std::uint64_t drain = 0;
  while (::read(fd, &drain, sizeof(drain)) > 0) {
    // Drain every pending wakeup tick — see EpollReactor::run()'s
    // identical wakeup_fd_ drain loop for why looping to EAGAIN (rather
    // than assuming one read is enough) is the only correct way to fully
    // drain an eventfd.
  }
  if (!completion_queue_.has_value()) {
    return;  // Can't happen (this callback is only ever registered when
              // completion_queue_ was just emplaced), but cheap to guard.
  }
  std::vector<concurrency::BoundedWorkQueue::Task> completions = completion_queue_->drain();
  for (auto& completion : completions) {
    completion();  // Each is an on_worker_result(...) call — see
                     // dispatch_to_worker() for how it was constructed.
  }
}

void NonblockingHttpServer::process_readable(const std::shared_ptr<HttpConnection>& conn_ptr) {
  HttpConnection& conn = *conn_ptr;
  std::vector<char> buffer(config_.read_chunk_bytes);
  for (;;) {
    net::IoResult result = net::read_some(conn.base.fd(), buffer.data(), buffer.size());
    if (result.bytes_transferred > 0) {
      if (!conn.base.read_buffer.append(buffer.data(), result.bytes_transferred)) {
        // read_buffer's own cap (docs/decisions/0006) was hit — see
        // EchoReactorServer::process_readable's identical handling.
        close_connection(conn.base.fd());
        return;
      }
      conn.base.touch();
    }
    if (result.status == net::IoStatus::Closed || result.status == net::IoStatus::Error) {
      close_connection(conn.base.fd());
      return;
    }
    if (result.status == net::IoStatus::WouldBlock || result.bytes_transferred < buffer.size()) {
      break;  // See EchoReactorServer::process_readable's identical comment.
    }
  }

  if (conn.awaiting_worker_result) {
    // A request on this connection is already dispatched to the worker
    // pool; whatever just arrived stays in read_buffer until
    // on_worker_result() resumes parsing — see this class's "Worker-pool
    // dispatch mode" docs for why (response ordering).
    return;
  }
  drive_parser(conn_ptr);
}

void NonblockingHttpServer::drive_parser(const std::shared_ptr<HttpConnection>& conn_ptr) {
  HttpConnection& conn = *conn_ptr;
  std::string_view view = conn.base.read_buffer.view();
  bool close_after_flush = false;
  // Set the instant the worker-pool branch below is taken, regardless of
  // whether dispatch_to_worker() ends up submitting asynchronously (still
  // pending — on_worker_result() will call finish_dispatch() itself, later,
  // once a worker completes) or resolving synchronously (its queue-full 503
  // fallback calls on_worker_result() -> finish_dispatch() immediately,
  // still nested inside this same call). Either way, finish_dispatch() must
  // never be called again from *this* function for this invocation: doing
  // so unconditionally used to be a real bug (see the comment at this
  // flag's use below).
  bool dispatched_to_worker_pool = false;

  for (;;) {
    protocol::ParseStatus status = conn.parser.feed(view);

    if (status == protocol::ParseStatus::kNeedMoreData) {
      break;  // `view` was fully consumed into the parser's internal state;
               // wait for the next readable event.
    }

    if (status == protocol::ParseStatus::kError) {
      // Best-effort: if the output queue happens to already be at capacity
      // (docs/decisions/0006), there is nowhere to put this error response
      // either — enqueue_output()'s false return is silently accepted here
      // (nothing new gets queued) and the connection still closes once
      // whatever *is* already queued has drained, same as the success case.
      [[maybe_unused]] bool queued =
          conn.base.enqueue_output(conn.parser.error_response().serialize());
      close_after_flush = true;
      view = std::string_view();  // discard any trailing bytes; closing regardless
      break;
    }

    // kComplete.
    if (config_.worker_pool != nullptr) {
      // Dispatch to the worker pool instead of calling handler_() inline
      // — see "Worker-pool dispatch mode" in this class's docs. `view`'s
      // current contents (any bytes belonging to a next pipelined request
      // already delivered in this same read) are deliberately left
      // untouched: they are preserved in read_buffer by the consume() call
      // below and re-parsed once this request's response comes back and
      // resumes drive_parser().
      protocol::HttpRequest request = std::move(conn.parser.request());
      bool request_keep_alive = request.keep_alive();
      conn.parser.reset();
      conn.awaiting_worker_result = true;
      dispatched_to_worker_pool = true;
      dispatch_to_worker(conn_ptr, std::move(request), request_keep_alive);
      break;
    }

    protocol::HttpResponse response = handler_(conn.parser.request());
    bool keep_alive = response.keep_alive && conn.parser.request().keep_alive();
    response.keep_alive = keep_alive;
    std::string serialized = response.serialize();
    if (!conn.base.enqueue_output(serialized)) {
      // The output queue is full (docs/decisions/0006): this connection's
      // peer isn't draining responses fast enough relative to how many it
      // keeps requesting. There is nowhere left to put this response, so
      // it is dropped and the connection is torn down once whatever *is*
      // already queued drains — bounded memory over a perfectly-formed
      // reply to this one request, per the spec's core design principle.
      // Phase 6 is where this becomes a configurable, observable policy
      // (e.g. a 503 sent proactively before the queue is actually full)
      // instead of a fixed "always close once it is" one.
      close_after_flush = true;
      view = std::string_view();
      break;
    }
    ++conn.requests_served;
    conn.parser.reset();

    if (!keep_alive || conn.requests_served >= config_.max_keep_alive_requests) {
      close_after_flush = true;
      break;
    }
    // Otherwise loop again: `view` may already hold the start of the next
    // pipelined request, delivered in the same recv(2) as this one.
  }

  // `view` is always a suffix of the bytes read_buffer held at the top of
  // this function (feed() only ever consumes a prefix via remove_prefix, or
  // this function discards the rest itself on kError/overflow above), so
  // read_buffer.size() - view.size() is exactly how many bytes were
  // consumed — see HttpRequestParser::feed()'s docs.
  conn.base.read_buffer.consume(conn.base.read_buffer.size() - view.size());

  if (dispatched_to_worker_pool) {
    // Checking conn.awaiting_worker_result here instead used to be the bug:
    // it is only true for the genuinely-still-pending (async submit())
    // case, but dispatch_to_worker()'s queue-full fallback resolves
    // synchronously by calling on_worker_result() (which resets
    // awaiting_worker_result to false and calls finish_dispatch() itself)
    // *before* returning control here. Guarding on conn.awaiting_worker_result
    // alone would then fall through to the unconditional finish_dispatch()
    // call below for that synchronous-fallback case — a second, redundant
    // flush_output() attempt on a connection finish_dispatch() had already
    // fully handled (possibly already closed). If that connection's output
    // still had unflushed bytes ahead of the just-enqueued response (e.g. a
    // slow-draining peer with a prior large response still in the kernel's
    // socket buffer, even though this connection's own OutputQueue had
    // already reported itself fully flushed), this redundant call's own
    // flush_output() could independently return kPending and stomp the
    // connection's state from the correct kClosing back to kWriting,
    // silently defeating a close the response had already promised the
    // client on the wire (see docs/decisions/0007's Consequences section).
    // dispatched_to_worker_pool covers both outcomes uniformly: whether
    // still pending (on_worker_result() will finish this later) or already
    // resolved synchronously (already finished), this function must not
    // call finish_dispatch() itself either way.
    return;
  }
  finish_dispatch(conn, close_after_flush);
}

void NonblockingHttpServer::dispatch_to_worker(const std::shared_ptr<HttpConnection>& conn_ptr,
                                                protocol::HttpRequest request,
                                                bool request_keep_alive) {
  std::weak_ptr<HttpConnection> weak_conn = conn_ptr;
  bool submitted = config_.worker_pool->submit(
      [this, weak_conn, request = std::move(request), request_keep_alive]() mutable {
        // Runs on a worker thread: handler_ must tolerate concurrent
        // invocation from multiple worker threads (see Config::worker_pool
        // docs) — this is the ONLY place a worker thread ever touches
        // anything from `this` besides handler_ and completion_queue_/
        // completion_fd_ (both internally synchronized), and it never
        // touches `conn_ptr`/`weak_conn`'s pointee at all — see class
        // docs for why. The inner lambda below is given its own,
        // distinctly-named capture of weak_conn (rather than recapturing
        // the same name) purely to keep -Wshadow happy about a nested
        // lambda capturing an identically-named enclosing capture.
        protocol::HttpResponse response = handler_(request);
        // Blocking push() here, not try_push(): this runs on a *worker*
        // thread, never the reactor thread, so waiting for space is safe —
        // unlike WorkerPool::submit() (called only from the reactor
        // thread), which must never block and rejects predictably instead.
        // A dropped completion here used to strand the connection forever:
        // conn.awaiting_worker_result would never be cleared (only
        // on_worker_result() clears it), so drive_parser() would refuse to
        // process this connection ever again — silent, permanent, and
        // reachable by ordinary configuration (completion_queue_capacity
        // set smaller than the worker pool's own concurrency), not just a
        // wedged pool. push() blocking until space frees (the reactor
        // thread drains completion_queue_ on every reactor loop iteration,
        // so this is normally near-instant) or the queue closes (see
        // run()'s completion_queue_->close(), which unblocks this during
        // shutdown instead of leaking a stuck worker thread) is what
        // guarantees this connection's response is always either delivered
        // or the connection is known to be shutting down — never silently
        // orphaned.
        bool queued = completion_queue_->push(
            [this, completion_weak_conn = weak_conn, response = std::move(response),
             request_keep_alive]() mutable {
              on_worker_result(completion_weak_conn, std::move(response), request_keep_alive);
            });
        if (queued) {
          std::uint64_t one = 1;
          (void)::write(completion_fd_.get(), &one, sizeof(one));
        }
        // !queued: completion_queue_ was closed (server shutting down, see
        // run()) while this worker was waiting — safe to drop: the class
        // docs' required shutdown ordering guarantees no caller is still
        // waiting on a live connection once run() has returned.
      });
  if (submitted) {
    return;
  }

  // The worker pool's own bounded queue is full (FR5: "policies for ... a
  // full worker queue") — reject predictably instead of blocking the
  // reactor thread or growing anything without bound. Reuses
  // on_worker_result() directly (synchronously, still on the reactor
  // thread — nothing was actually dispatched to another thread) so the
  // "enqueue response, decide close, resume parsing" bookkeeping isn't
  // duplicated.
  protocol::HttpResponse response = protocol::make_error_response(
      503, "Service Unavailable", "worker queue is full\n", /*keep_alive=*/false);
  on_worker_result(std::weak_ptr<HttpConnection>(conn_ptr), std::move(response),
                    request_keep_alive);
}

void NonblockingHttpServer::on_worker_result(std::weak_ptr<HttpConnection> conn_weak,
                                              protocol::HttpResponse response,
                                              bool request_keep_alive) {
  std::shared_ptr<HttpConnection> conn_ptr = conn_weak.lock();
  if (!conn_ptr) {
    // The connection was closed (client disconnected, or another error
    // path) while this request was being handled on a worker thread. Its
    // response has nowhere to go — expected and safe, not a bug: see this
    // class's "Worker-pool dispatch mode" docs for why a raw-fd-keyed
    // lookup here instead would have been unsafe (fd reuse).
    return;
  }
  HttpConnection& conn = *conn_ptr;
  conn.awaiting_worker_result = false;

  bool keep_alive = response.keep_alive && request_keep_alive;
  response.keep_alive = keep_alive;
  bool close_after_flush = true;
  if (conn.base.enqueue_output(response.serialize())) {
    ++conn.requests_served;
    close_after_flush = !keep_alive || conn.requests_served >= config_.max_keep_alive_requests;
  }
  // else: output queue full (docs/decisions/0006) — same bounded-overload
  // handling as drive_parser()'s inline path: drop this response, close
  // once whatever is already queued drains (close_after_flush stays true).

  bool resumable = finish_dispatch(conn, close_after_flush);
  if (resumable) {
    // More pipelined bytes may already be sitting in read_buffer, received
    // while this connection's one in-flight worker request was pending —
    // resume parsing them now that ordering is preserved.
    drive_parser(conn_ptr);
  }
}

bool NonblockingHttpServer::finish_dispatch(HttpConnection& conn, bool close_after_flush) {
  reactor::FlushResult flush_result = conn.base.flush_output();
  if (flush_result == reactor::FlushResult::kFailed) {
    close_connection(conn.base.fd());
    return false;
  }
  if (flush_result == reactor::FlushResult::kFlushed) {
    if (close_after_flush) {
      close_connection(conn.base.fd());
      return false;
    }
    return true;  // Stays kReading, interest already kReadable — safe for
                   // a caller to resume parsing immediately.
  }

  // kPending: the response(s) didn't fully fit in one send(2) call.
  conn.base.set_state(close_after_flush ? reactor::ConnectionState::kClosing
                                         : reactor::ConnectionState::kWriting);
  reactor_.modify(conn.base.fd(), reactor::kWritable);
  return false;  // Reading is paused while the write drains — not safe to
                  // resume parsing yet.
}

void NonblockingHttpServer::close_connection(int fd) noexcept {
  reactor_.remove(fd);
  connections_.erase(fd);
}

}  // namespace arcserve::server
