#include "arcserve/server/echo_reactor_server.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "test_socket_client.hpp"

namespace arcserve::server {
namespace {

using arcserve::testing::RawTcpClient;

// Runs an EchoReactorServer on a background thread for the lifetime of one
// test, mirroring BlockingHttpServerIntegrationTest's ServerFixture (Phase
// 1) — same reasoning applies to declaration order relative to any
// RawTcpClient in a test body: this fixture must be declared first so a
// client's socket is closed before stop()+join() runs.
class EchoServerFixture {
 public:
  EchoServerFixture() : server_(make_config()), thread_([this] { server_.run(); }) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ~EchoServerFixture() {
    server_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  EchoServerFixture(const EchoServerFixture&) = delete;
  EchoServerFixture& operator=(const EchoServerFixture&) = delete;

  [[nodiscard]] std::uint16_t port() const { return server_.port(); }
  [[nodiscard]] EchoReactorServer& server() { return server_; }

 private:
  static EchoReactorServer::Config make_config() {
    EchoReactorServer::Config config;
    config.port = 0;
    return config;
  }

  EchoReactorServer server_;
  std::thread thread_;
};

std::string read_exactly(RawTcpClient& client, std::size_t expected_size) {
  std::string data;
  while (data.size() < expected_size && !client.is_closed()) {
    data += client.read_chunk();
  }
  return data;
}

TEST(EchoReactorServerIntegrationTest, EchoesSingleMessage) {
  EchoServerFixture fixture;
  RawTcpClient client(fixture.port());

  const std::string message = "hello reactor";
  ASSERT_TRUE(client.send_fragmented(message));
  EXPECT_EQ(read_exactly(client, message.size()), message);
}

// FR3-equivalent property for the reactor's I/O path itself (not the HTTP
// parser, which Phase 3 covers separately): the reactor must correctly
// reassemble input delivered as many separate EPOLLIN-triggered reads, not
// just input that arrives in one recv(2) call.
TEST(EchoReactorServerIntegrationTest, EchoesByteFragmentedInput) {
  EchoServerFixture fixture;
  RawTcpClient client(fixture.port());

  const std::string message = "fragmented-one-byte-at-a-time";
  ASSERT_TRUE(client.send_fragmented(message, /*chunk_size=*/1));
  EXPECT_EQ(read_exactly(client, message.size()), message);
}

TEST(EchoReactorServerIntegrationTest, ClosingClientDoesNotCrashServer) {
  EchoServerFixture fixture;
  {
    RawTcpClient client(fixture.port());
    ASSERT_TRUE(client.send_fragmented("partial"));
    client.close_now();
  }

  RawTcpClient next(fixture.port());
  const std::string message = "still alive";
  ASSERT_TRUE(next.send_fragmented(message));
  EXPECT_EQ(read_exactly(next, message.size()), message);
}

// FR6 / partial-write path, exercised for the first time in this codebase:
// Phase 1's blocking write_all() is always allowed to block until fully
// sent, so it never needed EPOLLOUT-driven resumption. A payload larger
// than the OS's default socket buffers guarantees the server's write_all()
// call hits WouldBlock at least once, forcing EchoReactorServer into
// ConnectionState::kWriting and a later EPOLLOUT event to finish draining
// — this proves that path preserves data exactly (no corruption, no
// reordering, no truncation) under real partial-write conditions. The
// reading loop runs concurrently with (not after) the send, so this test
// cannot deadlock against the server's "pause reads while writing" policy
// (docs/decisions/0005-reactor-connection-model.md, decision 3).
TEST(EchoReactorServerIntegrationTest, LargeTransferSurvivesPartialWrites) {
  EchoServerFixture fixture;
  RawTcpClient client(fixture.port());

  constexpr std::size_t kPayloadSize = 2 * 1024 * 1024;  // 2 MiB
  std::string payload(kPayloadSize, '\0');
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>('a' + (i % 26));  // detect reordering/corruption, not just size
  }

  std::thread sender([&client, &payload] { ASSERT_TRUE(client.send_fragmented(payload)); });

  std::string echoed;
  echoed.reserve(payload.size());
  while (echoed.size() < payload.size() && !client.is_closed()) {
    echoed += client.read_chunk(65536);
  }
  sender.join();

  ASSERT_EQ(echoed.size(), payload.size());
  EXPECT_EQ(echoed, payload);
}

// Phase 2 exit criterion, direct statement: "No blocking I/O in reactor".
// A large transfer in flight on one connection (backed up enough to need
// EPOLLOUT-driven draining — see the test above) must not delay service to
// a concurrently-connected, well-behaved client. If the reactor thread
// ever blocked on the slow connection's I/O, the fast client's round trip
// would be stuck behind the whole multi-megabyte transfer instead of
// completing in the time its own tiny payload takes.
TEST(EchoReactorServerIntegrationTest, LargeTransferDoesNotBlockOtherClients) {
  EchoServerFixture fixture;

  constexpr std::size_t kPayloadSize = 2 * 1024 * 1024;  // 2 MiB
  const std::string payload(kPayloadSize, 'z');

  RawTcpClient slow_client(fixture.port());
  std::thread sender(
      [&slow_client, &payload] { ASSERT_TRUE(slow_client.send_fragmented(payload)); });
  std::thread reader([&slow_client, &payload] {
    std::string echoed;
    echoed.reserve(payload.size());
    while (echoed.size() < payload.size() && !slow_client.is_closed()) {
      echoed += slow_client.read_chunk(65536);
    }
    EXPECT_EQ(echoed, payload);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let the transfer get underway

  auto start = std::chrono::steady_clock::now();
  RawTcpClient fast_client(fixture.port());
  ASSERT_TRUE(fast_client.send_fragmented("quick"));
  std::string fast_echo = read_exactly(fast_client, 5);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_EQ(fast_echo, "quick");
  EXPECT_LT(elapsed, std::chrono::milliseconds(1000))
      << "a large concurrent transfer on another connection must not delay this one";

  sender.join();
  reader.join();
}

// Phase 2 exit criterion: "echo under many connections" / "stress test
// passes". Correctness (every one of N concurrent clients gets exactly its
// own bytes back, no cross-talk) is the primary assertion; the bounded
// elapsed time is a smoke-test upper bound against a genuine hang/deadlock,
// not a performance claim (Phase 8 owns those, with committed data).
TEST(EchoReactorServerIntegrationTest, HandlesManySimultaneousConnectionsCorrectly) {
  EchoServerFixture fixture;
  constexpr int kClients = 200;

  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  threads.reserve(kClients);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kClients; ++i) {
    threads.emplace_back([&fixture, &success_count, i] {
      RawTcpClient client(fixture.port());
      std::string message = "client-" + std::to_string(i) + "-payload";
      if (!client.send_fragmented(message)) {
        return;
      }
      std::string echoed = read_exactly(client, message.size());
      if (echoed == message) {
        success_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_EQ(success_count.load(), kClients);
  EXPECT_EQ(fixture.server().connections_accepted(), static_cast<std::size_t>(kClients));
  EXPECT_LT(elapsed, std::chrono::seconds(10));
}

}  // namespace
}  // namespace arcserve::server
