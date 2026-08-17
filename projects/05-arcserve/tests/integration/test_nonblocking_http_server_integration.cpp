#include "arcserve/server/nonblocking_http_server.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "arcserve/protocol/http_parser.hpp"
#include "arcserve/server/default_handlers.hpp"
#include "test_socket_client.hpp"

namespace arcserve::server {
namespace {

using arcserve::testing::RawTcpClient;
using arcserve::testing::read_one_http_response;

// Mirrors BlockingServerIntegrationTest's ServerFixture (Phase 1) exactly,
// swapped to NonblockingHttpServer — see that fixture's comment for why
// declaration order relative to a RawTcpClient matters.
class NonblockingServerFixture {
 public:
  NonblockingServerFixture()
      : server_(make_config(), default_route), thread_([this] { server_.run(); }) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ~NonblockingServerFixture() {
    server_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  NonblockingServerFixture(const NonblockingServerFixture&) = delete;
  NonblockingServerFixture& operator=(const NonblockingServerFixture&) = delete;

  [[nodiscard]] std::uint16_t port() const { return server_.port(); }

 private:
  static NonblockingHttpServer::Config make_config() {
    NonblockingHttpServer::Config config;
    config.port = 0;
    return config;
  }

  NonblockingHttpServer server_;
  std::thread thread_;
};

// A pipelining-aware sibling of arcserve::testing::read_one_http_response:
// that helper deliberately discards any bytes past the first response's
// declared Content-Length (see its comment in test_socket_client.cpp) —
// correct and sufficient for every other test in this suite, all of which
// send one request at a time, but not for
// PipelinedRequestsInOneWriteBothGetCorrectResponses below, which must
// recover *both* responses even when the server's single write(2) call
// delivers them to the client in the same recv(2). Reads and returns
// exactly two complete HTTP/1.1 responses from the raw byte stream.
std::pair<std::string, std::string> read_two_http_responses(RawTcpClient& client) {
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

  std::string first = extract_one();
  std::string second = extract_one();
  return {first, second};
}

// A throttled sibling of arcserve::testing::read_one_http_response: reads
// in small chunks with a short sleep between each recv(2) instead of
// draining as fast as possible. This is what actually exercises
// NonblockingHttpServer::try_flush()'s EPOLLOUT re-arm/drain path
// (ConnectionState::kWriting/kClosing): every other test in this file uses
// a response small enough to complete in a single send(2) call, so relying
// on payload size alone against OS socket buffers of unknown size (as
// EchoReactorServer's sibling tests do, at the raw-byte level) would not
// reliably force a partial write here. Deliberately reading slowly is the
// direct way to guarantee it, and is exactly the "slow client" scenario
// spec section 1.6 calls for.
std::string read_one_http_response_slowly(RawTcpClient& client) {
  static const std::string kTerminator = "\r\n\r\n";
  std::string buffer;

  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    std::string chunk = client.read_chunk(4096);
    if (chunk.empty() && client.is_closed()) {
      return buffer;
    }
    buffer += chunk;
    header_end = buffer.find(kTerminator);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    std::string chunk = client.read_chunk(4096);
    if (chunk.empty() && client.is_closed()) {
      break;
    }
    buffer += chunk;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (buffer.size() > needed) {
    buffer.resize(needed);
  }
  return buffer;
}

TEST(NonblockingHttpServerIntegrationTest, SimpleGetRequestRoundTrips) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find("ArcServe Phase 1 blocking server is alive."), std::string::npos);
}

TEST(NonblockingHttpServerIntegrationTest, PostEchoesBody) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string body = "hello reactor";
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find(body), std::string::npos);
}

TEST(NonblockingHttpServerIntegrationTest, UnknownRouteReturns404) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET /nope HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 404 Not Found"), std::string::npos);
}

// FR3 through the reactor: a request delivered one byte per recv() call —
// each byte its own separate EPOLLIN-triggered read_some() call, not just
// one feed() per application-level chunk as the parser unit tests exercise
// in-process.
TEST(NonblockingHttpServerIntegrationTest, FragmentedRequestArrivesCorrectly) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string body = "abcdefgh";
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request, /*chunk_size=*/1));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find(body), std::string::npos);
}

TEST(NonblockingHttpServerIntegrationTest, MalformedRequestGetsRejectedAndConnectionCloses) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("NOT_A_REQUEST_LINE\r\n\r\n"));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 400 Bad Request"), std::string::npos);

  std::string trailing = client.read_chunk();
  EXPECT_TRUE(trailing.empty());
  EXPECT_TRUE(client.is_closed());
}

TEST(NonblockingHttpServerIntegrationTest, OversizedBodyIsRejectedWith413WithoutSendingIt) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(protocol::HttpRequestParser::kMaxBodyBytes + 1) +
                         "\r\n\r\n";
  ASSERT_TRUE(client.send_fragmented(request));
  std::string response = read_one_http_response(client);

  EXPECT_NE(response.find("HTTP/1.1 413"), std::string::npos);
}

TEST(NonblockingHttpServerIntegrationTest, KeepAliveServesTwoRequestsOnOneConnection) {
  NonblockingServerFixture fixture;
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

TEST(NonblockingHttpServerIntegrationTest, ConnectionCloseHeaderClosesAfterOneResponse) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"));
  std::string response = read_one_http_response(client);
  EXPECT_NE(response.find("Connection: close"), std::string::npos);

  std::string trailing = client.read_chunk();
  EXPECT_TRUE(trailing.empty());
  EXPECT_TRUE(client.is_closed());
}

TEST(NonblockingHttpServerIntegrationTest, ClientDisconnectMidRequestDoesNotCrashServer) {
  NonblockingServerFixture fixture;
  {
    RawTcpClient client(fixture.port());
    ASSERT_TRUE(client.send_fragmented("POST /echo HTTP/1.1\r\nContent-Length: 100\r\n\r\n"));
    client.close_now();
  }

  RawTcpClient next_client(fixture.port());
  ASSERT_TRUE(next_client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(next_client);
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
}

// Two full keep-alive requests delivered in a single write() on the client
// side — a real pipelined burst, not the parser unit tests' in-process
// split. Proves drive_parser()'s inner loop resolves every complete
// request already sitting in one EPOLLIN-triggered read before waiting for
// the next reactor event, not just the first one.
TEST(NonblockingHttpServerIntegrationTest, PipelinedRequestsInOneWriteBothGetCorrectResponses) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  std::string combined =
      "GET / HTTP/1.1\r\nHost: test\r\n\r\n"
      "GET /nope HTTP/1.1\r\nHost: test\r\n\r\n";
  ASSERT_TRUE(client.send_fragmented(combined));

  auto [first, second] = read_two_http_responses(client);
  EXPECT_NE(first.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(second.find("HTTP/1.1 404 Not Found"), std::string::npos);
}

// try_flush()'s EPOLLOUT re-arm/drain path (the response didn't fit in one
// send(2) call) has zero coverage elsewhere in this file: every other test
// here uses a request/response small enough to complete in a single
// send(2), so drive_parser()'s "kPending" branch and
// ConnectionState::kWriting never run under test. The sibling
// EchoReactorServer has explicit coverage for the identical underlying
// send/EPOLLOUT mechanism, but not the HTTP-specific logic
// NonblockingHttpServer layers on top of it (keep-alive decided before the
// flush even starts, then resuming reads once it drains). Spec section 1.6
// explicitly requires "slow client" integration coverage; read_one_http_-
// response_slowly (a deliberately throttled reader) is what actually forces
// the partial write rather than merely hoping payload size exceeds
// whatever the OS's socket buffers happen to be.
TEST(NonblockingHttpServerIntegrationTest,
     LargeResponseBodySurvivesPartialWritesOnKeepAliveConnection) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  const std::size_t body_size = protocol::HttpRequestParser::kMaxBodyBytes;
  std::string body(body_size, '\0');
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<char>('a' + (i % 26));  // detects corruption/reordering, not just size
  }
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request));

  std::string response = read_one_http_response_slowly(client);
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  ASSERT_GE(response.size(), body.size());
  EXPECT_EQ(response.substr(response.size() - body.size()), body);

  // The connection must have returned to ConnectionState::kReading (not be
  // stuck in kWriting, and not have been torn down) once the partially-
  // written response finished draining: a second, ordinary request on the
  // same keep-alive connection must still succeed.
  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string second = read_one_http_response(client);
  EXPECT_NE(second.find("HTTP/1.1 200 OK"), std::string::npos);
}

// Sibling of the above, exercising the other branch try_flush()'s caller
// can leave a connection in: ConnectionState::kClosing (close_after_flush
// combined with a response that itself needs more than one send(2) call to
// drain) — reached here via "Connection: close", the same as
// ConnectionCloseHeaderClosesAfterOneResponse above but with a body large
// enough to force a genuine partial write first.
TEST(NonblockingHttpServerIntegrationTest, LargeResponseBodySurvivesPartialWritesThenCloses) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  const std::size_t body_size = protocol::HttpRequestParser::kMaxBodyBytes;
  std::string body(body_size, '\0');
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<char>('a' + (i % 26));
  }
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nConnection: close\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(client.send_fragmented(request));

  std::string response = read_one_http_response_slowly(client);
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find("Connection: close"), std::string::npos);
  ASSERT_GE(response.size(), body.size());
  EXPECT_EQ(response.substr(response.size() - body.size()), body);

  // close_after_flush + kClosing: the connection must actually close once
  // the partially-written response has fully drained, not stay open.
  std::string trailing = client.read_chunk();
  EXPECT_TRUE(trailing.empty());
  EXPECT_TRUE(client.is_closed());
}

// Phase 2's LargeTransferDoesNotBlockOtherClients, restated for HTTP: a
// connection whose response is large enough (and being drained slowly
// enough) to need several EPOLLOUT-driven partial writes must not delay
// service to a concurrently-connected, well-behaved client. If the reactor
// thread ever blocked on the slow connection's writes, the fast client's
// round trip would be stuck behind the whole multi-hundred-KB transfer
// instead of completing in the time its own tiny request/response takes.
TEST(NonblockingHttpServerIntegrationTest, SlowClientWithLargeResponseDoesNotBlockOtherClients) {
  NonblockingServerFixture fixture;

  const std::size_t body_size = protocol::HttpRequestParser::kMaxBodyBytes;
  const std::string body(body_size, 'z');
  std::string request = "POST /echo HTTP/1.1\r\nHost: test\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;

  RawTcpClient slow_client(fixture.port());
  ASSERT_TRUE(slow_client.send_fragmented(request));

  std::thread reader([&slow_client, &body] {
    std::string response = read_one_http_response_slowly(slow_client);
    EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
    ASSERT_GE(response.size(), body.size());
    EXPECT_EQ(response.substr(response.size() - body.size()), body);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let the slow response get underway

  auto start = std::chrono::steady_clock::now();
  RawTcpClient fast_client(fixture.port());
  ASSERT_TRUE(fast_client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string fast_response = read_one_http_response(fast_client);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_NE(fast_response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_LT(elapsed, std::chrono::milliseconds(1000))
      << "a large, slow-draining response on another connection must not delay this one";

  reader.join();
}

// Phase 3 exit criterion: "end-to-end requests succeed", exercised under
// the same concurrency conditions Phase 2's echo stress test proves for
// raw bytes — many simultaneous HTTP clients, none blocking on another.
TEST(NonblockingHttpServerIntegrationTest, HandlesManySimultaneousClientsCorrectly) {
  NonblockingServerFixture fixture;
  constexpr int kClients = 100;

  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  threads.reserve(kClients);

  for (int i = 0; i < kClients; ++i) {
    threads.emplace_back([&fixture, &success_count, i] {
      RawTcpClient client(fixture.port());
      std::string body = "payload-" + std::to_string(i);
      std::string request = "POST /echo HTTP/1.1\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\n\r\n" + body;
      if (!client.send_fragmented(request)) {
        return;
      }
      std::string response = read_one_http_response(client);
      if (response.find("HTTP/1.1 200 OK") != std::string::npos &&
          response.find(body) != std::string::npos) {
        success_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kClients);
}

// Phase 3 exit criterion: "Fuzz/property cases where practical". Property
// under test: for many pseudo-random byte strings sent as a "request", the
// server must never crash or wedge. No response-shape assertion is made
// per iteration — random bytes may or may not contain a CRLF sequence that
// completes a request line, so the server may reply with an error and
// close, or may simply still be waiting for more bytes that this test
// never sends (an incomplete request is not malformed, merely unfinished).
// Both are correct, non-crashing outcomes. A fixed seed keeps this
// reproducible rather than flaky; the health check after the loop is the
// concrete pass/fail signal for the property itself.
TEST(NonblockingHttpServerIntegrationTest, SurvivesManyRandomMalformedInputs) {
  NonblockingServerFixture fixture;
  std::mt19937 rng(0xA5C7);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> length_dist(1, 512);

  constexpr int kIterations = 50;
  for (int i = 0; i < kIterations; ++i) {
    RawTcpClient client(fixture.port());
    auto length = static_cast<std::size_t>(length_dist(rng));
    std::string garbage(length, '\0');
    for (char& c : garbage) {
      c = static_cast<char>(byte_dist(rng));
    }
    ASSERT_TRUE(client.send_fragmented(garbage)) << "iteration " << i;
  }

  // The server must still be fully healthy: able to accept a fresh
  // connection and serve a normal request correctly. A crashed or wedged
  // server fails this outright.
  RawTcpClient client(fixture.port());
  ASSERT_TRUE(client.send_fragmented("GET / HTTP/1.1\r\nHost: test\r\n\r\n"));
  std::string response = read_one_http_response(client);
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
}

// Deterministic sibling of the fuzz test above: a malformed request line
// that *does* resolve quickly (has a terminating CRLF) must still get the
// exact, documented error response and close the connection, same as
// Phase 1's blocking server.
TEST(NonblockingHttpServerIntegrationTest, DeterministicMalformedInputGetsExactErrorResponse) {
  NonblockingServerFixture fixture;
  RawTcpClient client(fixture.port());

  ASSERT_TRUE(client.send_fragmented("GET http://example.com/ HTTP/1.1\r\n\r\n"));
  std::string response = read_one_http_response(client);
  EXPECT_NE(response.find("HTTP/1.1 400 Bad Request"), std::string::npos);
}

}  // namespace
}  // namespace arcserve::server
