#include "arcserve/server/blocking_server.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "arcserve/protocol/http_parser.hpp"
#include "arcserve/server/default_handlers.hpp"
#include "test_socket_client.hpp"

namespace arcserve::server {
namespace {

using arcserve::testing::RawTcpClient;
using arcserve::testing::read_one_http_response;

// Runs a BlockingHttpServer on a background thread for the lifetime of one
// test: each test gets an isolated server on its own kernel-assigned port,
// and the server is always cleanly stopped and joined (RAII) even if an
// assertion fails partway through the test body.
//
// IMPORTANT for every test below: declare `ServerFixture` *before*
// `RawTcpClient` (as all of them do). Local variables destruct in reverse
// declaration order, so the client's socket closes first — which is what
// unblocks the server thread if it's sitting inside a blocking
// handle_connection() read — *before* the fixture's destructor calls
// stop() + join(). Reversing that order would risk join() not returning
// until the OS eventually notices the leaked socket, since (by design —
// see BlockingHttpServer's class comment, and
// ShutdownWaitsForInFlightKeepAliveConnectionToClose below)
// stop() cannot interrupt a connection already being served.
class ServerFixture {
 public:
  ServerFixture() : server_(make_config(), default_route), thread_([this] { server_.serve(); }) {
    // The listening socket is already bound (TcpListener::listen runs
    // synchronously in BlockingHttpServer's constructor, above), so
    // connect() would succeed immediately via the kernel's accept
    // backlog even without this. This just gives serve()'s poll loop a
    // head start so tests aren't waiting on the first poll_interval_ms
    // tick under slow/loaded CI.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  ~ServerFixture() {
    server_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  ServerFixture(const ServerFixture&) = delete;
  ServerFixture& operator=(const ServerFixture&) = delete;

  [[nodiscard]] std::uint16_t port() const { return server_.port(); }

 private:
  static BlockingHttpServer::Config make_config() {
    BlockingHttpServer::Config config;
    config.port = 0;  // ephemeral
    config.poll_interval_ms = 20;
    return config;
  }

  BlockingHttpServer server_;
  std::thread thread_;
};

TEST(BlockingServerIntegrationTest, SimpleGetRequestRoundTrips) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find("ArcServe Phase 1 blocking server is alive."), std::string::npos);
}

TEST(BlockingServerIntegrationTest, PostEchoesBody) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string body = "hello arcserve";
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find(body), std::string::npos);
}

TEST(BlockingServerIntegrationTest, UnknownRouteReturns404) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET /nope HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 404 Not Found"), std::string::npos);
}

// FR3 end to end: the parser must reassemble a request delivered one byte
// per recv() call, not just one byte per in-process feed() call (covered
// separately by the unit tests) — this drives real socket I/O.
TEST(BlockingServerIntegrationTest, FragmentedRequestArrivesCorrectly) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string body = "abcdefgh";
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request, /*chunk_size=*/1));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find(body), std::string::npos);
}

TEST(BlockingServerIntegrationTest, MalformedRequestGetsRejectedAndConnectionCloses) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  // No spaces at all: fails request-line *syntax* (400), as distinct from
  // a well-formed three-token line with an unrecognized method (which is
  // 501 — see HttpRequestParserTest.RejectsUnsupportedMethod).
  ASSERT_TRUE(client.send_fragmented("NOT_A_REQUEST_LINE\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 400 Bad Request"), std::string::npos);

  // Protocol errors always close the connection: the next read observes EOF.
  std::string trailing = client.read_chunk();
  EXPECT_TRUE(trailing.empty());
  EXPECT_TRUE(client.is_closed());
}

TEST(BlockingServerIntegrationTest, OversizedBodyIsRejectedWith413WithoutSendingIt) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  // Only the headers are sent — the parser must reject based on the
  // Content-Length value alone, before any body bytes exist to read.
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(protocol::HttpRequestParser::kMaxBodyBytes + 1) +
                         "\r\n\r\n";
  ASSERT_TRUE(client.send_fragmented(request));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 413"), std::string::npos);
}

TEST(BlockingServerIntegrationTest, KeepAliveServesTwoRequestsOnOneConnection) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string first = read_one_http_response(client);
  EXPECT_NE(first.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(first.find("Connection: keep-alive"), std::string::npos);
  EXPECT_FALSE(client.is_closed());

  ASSERT_TRUE(client.send_fragmented("GET /nope HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string second = read_one_http_response(client);
  EXPECT_NE(second.find("HTTP/1.1 404 Not Found"), std::string::npos);
}

TEST(BlockingServerIntegrationTest, ConnectionCloseHeaderClosesAfterOneResponse) {
  ServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(
      client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"));
  std::string response = read_one_http_response(client);
  EXPECT_NE(response.find("Connection: close"), std::string::npos);

  std::string trailing = client.read_chunk();
  EXPECT_TRUE(trailing.empty());
  EXPECT_TRUE(client.is_closed());
}

TEST(BlockingServerIntegrationTest, ClientDisconnectMidRequestDoesNotCrashServer) {
  ServerFixture fixture;
  {
    RawTcpClient client(fixture.port());
    // Announce a 100-byte body, then vanish before sending it.
    ASSERT_TRUE(client.send_fragmented("POST /echo HTTP/1.1\r\nContent-Length: 100\r\n\r\n"));
    client.close_now();
  }

  // The server must still be alive and able to serve a fresh connection —
  // a crash or a wedged accept loop would fail this instead.
  RawTcpClient next_client(fixture.port());
  ASSERT_TRUE(next_client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(next_client);
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
}

TEST(BlockingServerIntegrationTest, SequentialClientsAreAllServed) {
  ServerFixture fixture;
  for (int i = 0; i < 5; ++i) {
    RawTcpClient client(fixture.port());
    ASSERT_TRUE(
        client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"));
    std::string response = read_one_http_response(client);
    EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos) << "client " << i;
  }
}

// spec §1.6 Failure category: "shutdown while active". Exercises the
// documented invariant directly (BlockingHttpServer's class comment):
// stop() cannot interrupt a connection already being served — the server
// thread is blocked inside handle_connection()'s blocking read, not in
// serve()'s poll loop, until that connection ends. Built on a raw
// BlockingHttpServer (not ServerFixture) because this test needs to
// observe serve() *not yet* having returned, which ServerFixture's
// stop-then-join destructor deliberately hides.
TEST(BlockingServerIntegrationTest, ShutdownWaitsForInFlightKeepAliveConnectionToClose) {
  BlockingHttpServer::Config config;
  config.port = 0;
  config.poll_interval_ms = 20;
  BlockingHttpServer server(config, default_route);

  std::atomic<bool> serve_returned{false};
  std::thread server_thread([&] {
    server.serve();
    serve_returned.store(true, std::memory_order_relaxed);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let serve() reach its poll loop

  RawTcpClient client(server.port());
  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);
  ASSERT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  ASSERT_NE(response.find("Connection: keep-alive"), std::string::npos);
  // The connection is now idle-but-open: the server thread is parked in a
  // blocking read inside handle_connection(), waiting for either another
  // request or a close.

  server.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(150));  // several poll_interval_ms
  EXPECT_FALSE(serve_returned.load(std::memory_order_relaxed))
      << "stop() must not interrupt a connection already being served (Phase 1 has no idle "
         "timeout — that's Phase 6)";

  // Closing the idle connection is what finally unblocks the server's
  // blocking read; handle_connection() returns, and serve() then observes
  // stop_requested_ on its very next poll() iteration.
  client.close_now();
  server_thread.join();
  EXPECT_TRUE(serve_returned.load(std::memory_order_relaxed));
}

// spec: "stop() ... Requests serve() return promptly (within one
// poll_interval_ms)." Bounds that latency concretely for the common case
// (no connections in flight at all).
TEST(BlockingServerIntegrationTest, GracefulShutdownWithNoConnectionsIsPrompt) {
  auto start = std::chrono::steady_clock::now();
  { ServerFixture fixture; }  // construct, then immediately destruct (stop + join)
  auto elapsed = std::chrono::steady_clock::now() - start;

  // Generous margin over ServerFixture's poll_interval_ms (20ms) plus its
  // own 20ms startup grace period, to stay robust under loaded CI.
  EXPECT_LT(elapsed, std::chrono::milliseconds(1000));
}

}  // namespace
}  // namespace arcserve::server
