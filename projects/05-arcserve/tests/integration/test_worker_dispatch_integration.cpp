#include "arcserve/concurrency/worker_pool.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "arcserve/protocol/http_message.hpp"
#include "arcserve/server/default_handlers.hpp"
#include "arcserve/server/nonblocking_http_server.hpp"
#include "test_socket_client.hpp"

namespace arcserve::server {
namespace {

using arcserve::testing::RawTcpClient;
using arcserve::testing::read_one_http_response;

// Reads exactly `count` complete HTTP/1.1 responses off `client`, in
// order, off a raw byte stream where more than one response may have
// arrived in the same recv(2) — unlike
// arcserve::testing::read_one_http_response (which deliberately discards
// any bytes past its one response's Content-Length; fine for every test
// that sends one request at a time, wrong here). Mirrors
// test_nonblocking_http_server_integration.cpp's identical local helper.
std::vector<std::string> read_n_http_responses(RawTcpClient& client, int count) {
  static const std::string kTerminator = "\r\n\r\n";
  std::string buffer;

  auto extract_one = [&]() -> std::string {
    std::size_t header_end = buffer.find(kTerminator);
    while (header_end == std::string::npos) {
      std::string chunk = client.read_chunk();
      if (chunk.empty() && client.is_closed()) {
        return "";
      }
      buffer += chunk;
      header_end = buffer.find(kTerminator);
    }

    std::size_t content_length = 0;
    std::string headers = buffer.substr(0, header_end);
    static const std::string kKey = "Content-Length:";
    std::size_t pos = headers.find(kKey);
    if (pos != std::string::npos) {
      pos += kKey.size();
      while (pos < headers.size() && headers[pos] == ' ') {
        ++pos;
      }
      std::size_t end = headers.find("\r\n", pos);
      if (end == std::string::npos) {
        end = headers.size();
      }
      std::from_chars(headers.data() + pos, headers.data() + end, content_length);
    }

    std::size_t needed = header_end + kTerminator.size() + content_length;
    while (buffer.size() < needed) {
      std::string chunk = client.read_chunk();
      if (chunk.empty() && client.is_closed()) {
        break;
      }
      buffer += chunk;
    }

    std::string response = buffer.substr(0, std::min(needed, buffer.size()));
    buffer.erase(0, response.size());
    return response;
  };

  std::vector<std::string> responses;
  responses.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    responses.push_back(extract_one());
  }
  return responses;
}

protocol::HttpResponse ok_response(std::string_view body) {
  protocol::HttpResponse resp;
  resp.status_code = 200;
  resp.reason_phrase = "OK";
  resp.body = std::string(body);
  return resp;
}

// Burns CPU (never yields, never blocks on I/O) for roughly `duration` —
// deliberately unlike an I/O wait, so running this inline on the reactor
// thread genuinely occupies it, not just "looks slow from the outside".
protocol::HttpResponse busy_wait_handler(const protocol::HttpRequest&,
                                          std::chrono::milliseconds duration) {
  auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    // Busy-spin: deliberately never yields or blocks on I/O, so running
    // this inline on the reactor thread genuinely occupies it for the
    // whole duration — see CpuHeavyHandlerBlocksReactorWithoutWorkerPool.
  }
  return ok_response("done");
}

// A small owning fixture: server + the worker pool it dispatches to (when
// `pool_config` is provided) + the background thread driving run(). Follows
// exactly the shutdown ordering NonblockingHttpServer's "Worker-pool
// dispatch mode" docs require: server stopped and joined first, *then* the
// pool stopped — see this fixture's destructor.
class WorkerDispatchFixture {
 public:
  explicit WorkerDispatchFixture(RequestHandler handler,
                                  concurrency::WorkerPool::Config pool_config = {})
      : worker_pool_(pool_config),
        server_(build_config(), std::move(handler)),
        thread_([this] { server_.run(); }) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  ~WorkerDispatchFixture() {
    server_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
    // Only after the reactor thread has fully stopped (so no *new* work
    // can be submitted) do we join the pool's worker threads — see
    // NonblockingHttpServer's class docs for why this order is required,
    // not just tidy.
    worker_pool_.stop();
  }

  WorkerDispatchFixture(const WorkerDispatchFixture&) = delete;
  WorkerDispatchFixture& operator=(const WorkerDispatchFixture&) = delete;

  [[nodiscard]] std::uint16_t port() const { return server_.port(); }
  [[nodiscard]] concurrency::WorkerPool& worker_pool() { return worker_pool_; }

 private:
  NonblockingHttpServer::Config build_config() {
    NonblockingHttpServer::Config config;
    config.port = 0;
    config.worker_pool = &worker_pool_;
    return config;
  }

  concurrency::WorkerPool worker_pool_;
  NonblockingHttpServer server_;
  std::thread thread_;
};

TEST(WorkerDispatchIntegrationTest, SimpleRequestThroughWorkerPoolGetsCorrectResponse) {
  WorkerDispatchFixture fixture(default_route);
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find("ArcServe Phase 1 blocking server is alive."), std::string::npos);
}

TEST(WorkerDispatchIntegrationTest, KeepAliveServesSequentialRequestsThroughWorkerPool) {
  WorkerDispatchFixture fixture(default_route);
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string first = read_one_http_response(client);
  EXPECT_NE(first.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_FALSE(client.is_closed());

  ASSERT_TRUE(client.send_fragmented("GET /nope HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string second = read_one_http_response(client);
  EXPECT_NE(second.find("HTTP/1.1 404 Not Found"), std::string::npos);
}

// The correctness property that makes worker-pool dispatch safe for a
// pipelined keep-alive connection at all: responses must come back in
// request order even though worker threads may finish out of submission
// order. The handler here deliberately delays the *first* request longer
// than the second — an adversarial ordering that would flip the responses
// if NonblockingHttpServer didn't hold each connection to one in-flight
// worker request at a time (see its "Worker-pool dispatch mode" docs).
TEST(WorkerDispatchIntegrationTest, PipelinedRequestsPreserveOrderDespiteVaryingHandlerDelay) {
  auto handler = [](const protocol::HttpRequest& request) {
    if (request.body == "first") {
      std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
    return ok_response(request.body);
  };
  WorkerDispatchFixture fixture(handler, concurrency::WorkerPool::Config{.worker_count = 4});
  RawTcpClient client(fixture.port());

  auto make_post = [](std::string_view body) {
    return "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
           std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
  };
  std::string combined = make_post("first") + make_post("second");
  ASSERT_TRUE(client.send_fragmented(combined));

  std::vector<std::string> responses = read_n_http_responses(client, 2);
  ASSERT_EQ(responses.size(), 2u);
  EXPECT_NE(responses[0].find("first"), std::string::npos)
      << "response 1 must be for the request sent first, even though its handler took longer";
  EXPECT_NE(responses[1].find("second"), std::string::npos);
}

// The handler shared by both comparison tests below: busy-waits only for
// requests to "/slow", and answers every other request instantly. This
// distinction is what makes the comparison meaningful — a fast_client
// sending a *different* (cheap) request must not be penalized just because
// *some* connection somewhere is running a slow handler; what's under test
// is specifically whether the reactor thread itself is available to
// service fast_client promptly while slow_client's request executes.
protocol::HttpResponse slow_path_handler(const protocol::HttpRequest& request) {
  if (request.target == "/slow") {
    return busy_wait_handler(request, std::chrono::milliseconds(150));
  }
  return ok_response("fast");
}

TEST(WorkerDispatchIntegrationTest, CpuHeavyHandlerBlocksReactorWithoutWorkerPool) {
  NonblockingHttpServer::Config config;
  config.port = 0;
  NonblockingHttpServer server(config, slow_path_handler);
  std::thread thread([&server] { server.run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  RawTcpClient slow_client(server.port());
  ASSERT_TRUE(slow_client.send_fragmented("GET /slow HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let the reactor start the busy handler

  auto start = std::chrono::steady_clock::now();
  RawTcpClient fast_client(server.port());
  ASSERT_TRUE(fast_client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string fast_response = read_one_http_response(fast_client);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_NE(fast_response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_GT(elapsed, std::chrono::milliseconds(100))
      << "baseline (no worker pool): a CPU-heavy handler for a DIFFERENT connection's request "
         "SHOULD still delay this one, since both run inline on the same single reactor thread";

  server.stop();
  thread.join();
}

// Direct-reactor-vs-worker-dispatch comparison (Phase 5 roadmap item),
// restated as a bounded pass/fail correctness assertion rather than a
// benchmark claim (spec rule 6: no performance numbers without committed
// raw data — that's Phase 8's job): the identical CPU-heavy handler
// (running for a *different* connection's request), this time dispatched
// to a worker pool, must NOT measurably delay a concurrently-connected
// fast client's *different, cheap* request.
TEST(WorkerDispatchIntegrationTest, CpuHeavyHandlerDoesNotBlockReactorWithWorkerPool) {
  WorkerDispatchFixture fixture(slow_path_handler, concurrency::WorkerPool::Config{.worker_count = 2});

  RawTcpClient slow_client(fixture.port());
  ASSERT_TRUE(slow_client.send_fragmented("GET /slow HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let the worker start the busy handler

  auto start = std::chrono::steady_clock::now();
  RawTcpClient fast_client(fixture.port());
  ASSERT_TRUE(fast_client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string fast_response = read_one_http_response(fast_client);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_NE(fast_response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_LT(elapsed, std::chrono::milliseconds(80))
      << "with a worker pool configured: a CPU-heavy handler on another connection must NOT "
         "delay this one — the reactor thread only ever submits the task, never runs it";

  std::string slow_response = read_one_http_response(slow_client);
  EXPECT_NE(slow_response.find("HTTP/1.1 200 OK"), std::string::npos);
}

// Phase 5 exit criterion: "saturation tests pass". A worker pool with
// exactly one worker and a queue capacity of one: the worker is kept busy
// by one connection's request, a second connection fills the (size-1)
// queue, and a third connection's request must be rejected predictably —
// a 503, not a hang, a crash, or unbounded growth — matching FR5 "policies
// for ... a full worker queue" and the spec's core design principle.
TEST(WorkerDispatchIntegrationTest, WorkerQueueSaturationRejectsExcessRequestsAndServerSurvives) {
  std::mutex release_mutex;
  std::condition_variable release_cv;
  bool released = false;
  std::atomic<int> handler_invocations{0};

  auto blocking_handler = [&](const protocol::HttpRequest&) {
    handler_invocations.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lock(release_mutex);
    release_cv.wait(lock, [&released] { return released; });
    return ok_response("released");
  };

  WorkerDispatchFixture fixture(blocking_handler,
                                 concurrency::WorkerPool::Config{.worker_count = 1, .queue_capacity = 1});

  RawTcpClient client_a(fixture.port());
  ASSERT_TRUE(client_a.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  for (int i = 0; i < 2000 && handler_invocations.load() < 1; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GE(handler_invocations.load(), 1) << "client A's request never reached the handler";

  RawTcpClient client_b(fixture.port());
  ASSERT_TRUE(client_b.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let it fill the 1-slot queue

  RawTcpClient client_c(fixture.port());
  ASSERT_TRUE(client_c.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response_c = read_one_http_response(client_c);
  EXPECT_NE(response_c.find("HTTP/1.1 503"), std::string::npos);

  {
    std::lock_guard<std::mutex> lock(release_mutex);
    released = true;
  }
  release_cv.notify_all();

  std::string response_a = read_one_http_response(client_a);
  EXPECT_NE(response_a.find("HTTP/1.1 200"), std::string::npos);
  std::string response_b = read_one_http_response(client_b);
  EXPECT_NE(response_b.find("HTTP/1.1 200"), std::string::npos);

  // The server must still be fully healthy once the overload clears.
  RawTcpClient client_d(fixture.port());
  ASSERT_TRUE(client_d.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response_d = read_one_http_response(client_d);
  EXPECT_NE(response_d.find("HTTP/1.1 200"), std::string::npos);
}

// Phase 5 exit criterion: "shutdown ... tests pass", with genuine
// in-flight work (unlike WorkerPoolTest's own shutdown tests, this
// exercises the full server+pool integration, including connections whose
// response may never get flushed because the reactor already stopped —
// see WorkerDispatchFixture's destructor for the ordering this proves
// doesn't hang or crash).
TEST(WorkerDispatchIntegrationTest, ServerAndPoolShutdownIsCleanWithWorkInFlight) {
  auto handler = [](const protocol::HttpRequest&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return ok_response("ok");
  };

  {
    WorkerDispatchFixture fixture(handler, concurrency::WorkerPool::Config{.worker_count = 2});

    std::vector<std::unique_ptr<RawTcpClient>> clients;
    for (int i = 0; i < 5; ++i) {
      auto client = std::make_unique<RawTcpClient>(fixture.port());
      ASSERT_TRUE(client->send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
      clients.push_back(std::move(client));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));  // let some requests reach the pool
    // Fixture teardown (below) stops the server, joins its thread, then
    // stops the pool — with several requests still in flight. Must not
    // hang or crash.
  }
  SUCCEED();
}

// Reads one HTTP/1.1 response the same way read_one_http_response() does,
// except bounded by a wall-clock deadline instead of blocking forever: `client`
// must have been constructed with a nonzero recv_timeout so each individual
// read_chunk() call itself times out and returns "" (not treated as closed)
// rather than blocking indefinitely. Exists specifically for
// SaturatedCompletionChannelNeverStrandsAConnection below, a regression test
// for a bug where a connection could be silently and permanently stranded —
// using the ordinary (unbounded) read_one_http_response() there would turn a
// regression into a hung test binary instead of a failed assertion.
std::string read_one_http_response_before_deadline(RawTcpClient& client,
                                                     std::chrono::steady_clock::time_point deadline) {
  static const std::string kTerminator = "\r\n\r\n";
  std::string buffer;

  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return buffer;
    }
    std::string chunk = client.read_chunk();
    if (chunk.empty() && client.is_closed()) {
      return buffer;
    }
    buffer += chunk;
    header_end = buffer.find(kTerminator);
  }

  std::size_t content_length = 0;
  {
    std::string headers = buffer.substr(0, header_end);
    static const std::string kKey = "Content-Length:";
    std::size_t pos = headers.find(kKey);
    if (pos != std::string::npos) {
      pos += kKey.size();
      while (pos < headers.size() && headers[pos] == ' ') {
        ++pos;
      }
      std::size_t end = headers.find("\r\n", pos);
      if (end == std::string::npos) {
        end = headers.size();
      }
      std::from_chars(headers.data() + pos, headers.data() + end, content_length);
    }
  }

  std::size_t needed = header_end + kTerminator.size() + content_length;
  while (buffer.size() < needed) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return buffer;
    }
    std::string chunk = client.read_chunk();
    if (chunk.empty() && client.is_closed()) {
      break;
    }
    buffer += chunk;
  }

  if (buffer.size() > needed) {
    buffer.resize(needed);
  }
  return buffer;
}

// Regression test (docs/decisions/0007-worker-pool-sizing.md's amendment):
// completion_queue_ — the channel a worker thread posts its finished
// HttpResponse through, independent of the worker pool's own task queue —
// used to drop a response outright if it happened to be full at the moment a
// worker finished, permanently stranding that connection: no response, no
// close, awaiting_worker_result stuck true forever with nothing left to ever
// clear it. A deliberately tiny completion_queue_capacity relative to
// worker_count makes many workers race to post completions through a
// channel that's almost always full; every one of these concurrent clients
// must still eventually get a real response — completion_queue_->push() now
// blocks the *worker* thread posting the completion (never the reactor
// thread) until the reactor thread catches up, rather than dropping
// anything.
TEST(WorkerDispatchIntegrationTest, SaturatedCompletionChannelNeverStrandsAConnection) {
  concurrency::WorkerPool worker_pool(
      concurrency::WorkerPool::Config{.worker_count = 8, .queue_capacity = 64});

  auto handler = [](const protocol::HttpRequest&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return ok_response("done");
  };

  NonblockingHttpServer::Config config;
  config.port = 0;
  config.worker_pool = &worker_pool;
  config.completion_queue_capacity = 1;  // deliberately tiny: force the race
  NonblockingHttpServer server(config, handler);
  std::thread server_thread([&server] { server.run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  constexpr int kClients = 20;
  std::vector<std::thread> clients;
  std::atomic<int> success_count{0};
  clients.reserve(kClients);
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([&server, &success_count] {
      RawTcpClient client(server.port(), std::chrono::milliseconds(200));
      if (!client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n")) {
        return;
      }
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      std::string response = read_one_http_response_before_deadline(client, deadline);
      if (response.find("HTTP/1.1 200 OK") != std::string::npos) {
        success_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : clients) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kClients)
      << "every client must eventually get a response even when the completion "
         "channel is saturated by many workers finishing near-simultaneously — "
         "none may be silently and permanently stranded";

  server.stop();
  server_thread.join();
  worker_pool.stop();
}

TEST(WorkerDispatchIntegrationTest, ManySimultaneousClientsThroughWorkerPoolAllSucceed) {
  WorkerDispatchFixture fixture(default_route, concurrency::WorkerPool::Config{.worker_count = 4});
  constexpr int kClients = 50;

  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  threads.reserve(kClients);

  for (int i = 0; i < kClients; ++i) {
    threads.emplace_back([&fixture, &success_count] {
      RawTcpClient client(fixture.port());
      if (!client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n")) {
        return;
      }
      std::string response = read_one_http_response(client);
      if (response.find("HTTP/1.1 200 OK") != std::string::npos) {
        success_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kClients);
}

}  // namespace
}  // namespace arcserve::server
