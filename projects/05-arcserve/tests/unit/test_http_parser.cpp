#include "arcserve/protocol/http_parser.hpp"

#include <cstddef>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

namespace arcserve::protocol {
namespace {

TEST(HttpRequestParserTest, ParsesSimpleGetRequest) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);

  EXPECT_EQ(parser.request().method, HttpMethod::kGet);
  EXPECT_EQ(parser.request().target, "/");
  EXPECT_EQ(parser.request().version, "HTTP/1.1");
  EXPECT_TRUE(parser.request().body.empty());

  auto host = find_header(parser.request().headers, "Host");
  ASSERT_TRUE(host.has_value());
  EXPECT_EQ(*host, "localhost");
  EXPECT_TRUE(data.empty());  // exactly one request was in the buffer
}

TEST(HttpRequestParserTest, ParsesPostWithBody) {
  HttpRequestParser parser;
  std::string_view data = "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);

  EXPECT_EQ(parser.request().method, HttpMethod::kPost);
  EXPECT_EQ(parser.request().target, "/echo");
  EXPECT_EQ(parser.request().body, "hello");
}

TEST(HttpRequestParserTest, EmptyContentLengthMeansNoBody) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_TRUE(parser.request().body.empty());
}

// FR3: incremental parser must handle fragmented input. Feeding one byte
// per feed() call is the strictest possible test of that property.
TEST(HttpRequestParserTest, HandlesByteAtATimeFragmentation) {
  const std::string full = "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
  HttpRequestParser parser;

  ParseStatus status = ParseStatus::kNeedMoreData;
  for (std::size_t i = 0; i < full.size(); ++i) {
    std::string_view one_byte(full.data() + i, 1);
    status = parser.feed(one_byte);
    if (i + 1 < full.size()) {
      ASSERT_EQ(status, ParseStatus::kNeedMoreData) << "unexpected status at byte index " << i;
    }
  }
  ASSERT_EQ(status, ParseStatus::kComplete);
  EXPECT_EQ(parser.request().method, HttpMethod::kPost);
  EXPECT_EQ(parser.request().target, "/echo");
  EXPECT_EQ(parser.request().body, "hello");
}

// Split the same request at every possible two-piece boundary: covers
// splits inside the request line, inside a header line, across the
// header/body boundary, and inside the body.
TEST(HttpRequestParserTest, HandlesSplitAtEveryOffset) {
  const std::string full = "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\nabc";

  for (std::size_t split = 1; split < full.size(); ++split) {
    HttpRequestParser parser;
    std::string_view first(full.data(), split);
    std::string_view second(full.data() + split, full.size() - split);

    ParseStatus status = parser.feed(first);
    if (status != ParseStatus::kComplete) {
      ASSERT_EQ(status, ParseStatus::kNeedMoreData) << "split at " << split;
      status = parser.feed(second);
    }
    ASSERT_EQ(status, ParseStatus::kComplete) << "split at " << split;
    EXPECT_EQ(parser.request().target, "/echo") << "split at " << split;
    EXPECT_EQ(parser.request().body, "abc") << "split at " << split;
  }
}

TEST(HttpRequestParserTest, RejectsUnsupportedMethod) {
  HttpRequestParser parser;
  std::string_view data = "DELETE / HTTP/1.1\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 501);
}

TEST(HttpRequestParserTest, RejectsLowercaseMethod) {
  HttpRequestParser parser;
  std::string_view data = "get / HTTP/1.1\r\n\r\n";  // HTTP methods are case-sensitive tokens
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 501);
}

TEST(HttpRequestParserTest, RejectsNonOriginFormTarget) {
  HttpRequestParser parser;
  std::string_view data = "GET http://example.com/ HTTP/1.1\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, RejectsUnsupportedVersion) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.0\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 505);
}

TEST(HttpRequestParserTest, RejectsMissingContentLengthOnPost) {
  HttpRequestParser parser;
  std::string_view data = "POST /echo HTTP/1.1\r\nHost: x\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 411);
}

TEST(HttpRequestParserTest, RejectsInvalidContentLengthValue) {
  HttpRequestParser parser;
  std::string_view data = "POST /echo HTTP/1.1\r\nContent-Length: not-a-number\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, RejectsOversizedBodyWithoutWaitingForIt) {
  HttpRequestParser parser;
  std::string request = "POST /echo HTTP/1.1\r\nContent-Length: " +
                         std::to_string(HttpRequestParser::kMaxBodyBytes + 1) + "\r\n\r\n";
  std::string_view data = request;
  // The parser must fail as soon as it sees the oversized Content-Length
  // header — before any body bytes arrive.
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 413);
}

TEST(HttpRequestParserTest, RejectsMalformedRequestLine) {
  HttpRequestParser parser;
  std::string_view data = "NOT_A_REQUEST_LINE\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, RejectsOversizedRequestLine) {
  HttpRequestParser parser;
  std::string oversized_target(HttpRequestParser::kMaxRequestLineLength + 10, 'a');
  std::string request = "GET /" + oversized_target + " HTTP/1.1\r\n\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, RejectsTooManyHeaders) {
  HttpRequestParser parser;
  std::string request = "GET / HTTP/1.1\r\n";
  for (int i = 0; i < 200; ++i) {
    request += "X-Test-" + std::to_string(i) + ": v\r\n";
  }
  request += "\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 431);
}

// Boundary regression test for the header-byte cap. A single header line
// whose raw content is exactly kMaxHeaderBytes long (7-byte "X-Pad: "
// prefix + padding) totals kMaxHeaderBytes + 2 bytes on the wire once its
// CRLF is counted — 2 bytes over the documented cap. The buggy version of
// this check compared header_bytes_seen_ + line.size() (no CRLF) against
// the cap, so it wrongly accepted this line.
TEST(HttpRequestParserTest, RejectsHeaderLineThatIsOnlyOverCapOnceItsCrlfCounts) {
  HttpRequestParser parser;
  constexpr std::size_t kPrefixLen = std::string_view("X-Pad: ").size();
  const std::string padding(HttpRequestParser::kMaxHeaderBytes - kPrefixLen, 'a');
  std::string request = "GET / HTTP/1.1\r\nX-Pad: " + padding + "\r\n\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 431);
}

// The exact-at-cap sibling of the above: a header line whose full wire
// bytes (content + CRLF) total precisely kMaxHeaderBytes must still be
// accepted — the fix must not become an off-by-one in the other direction.
TEST(HttpRequestParserTest, AcceptsHeaderBytesExactlyAtCap) {
  HttpRequestParser parser;
  constexpr std::size_t kPrefixLen = std::string_view("X-Pad: ").size();
  const std::string padding(HttpRequestParser::kMaxHeaderBytes - 2 - kPrefixLen, 'a');
  std::string request = "GET / HTTP/1.1\r\nX-Pad: " + padding + "\r\n\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
}

// Same exact-at-cap header block as above, but the header-terminating blank
// line arrives in a separate feed() call from the header that fills the
// cap — and, in the second variant, its own CRLF is itself split across
// feed() calls. Regression: parse_headers()'s early-out size check used to
// charge the terminator's own CRLF bytes against header_bytes_seen_ before
// consume_header_line() ever got a chance to recognize it as the (exempt)
// empty terminator line, so a request that was accepted in one feed() call
// was wrongly rejected once the terminator was split into its own call(s).
TEST(HttpRequestParserTest, AcceptsHeaderBytesExactlyAtCapWhenTerminatorArrivesSeparately) {
  constexpr std::size_t kPrefixLen = std::string_view("X-Pad: ").size();
  const std::string padding(HttpRequestParser::kMaxHeaderBytes - 2 - kPrefixLen, 'a');
  const std::string head = "GET / HTTP/1.1\r\nX-Pad: " + padding + "\r\n";

  {
    HttpRequestParser parser;
    std::string_view first = head;
    ASSERT_EQ(parser.feed(first), ParseStatus::kNeedMoreData);
    std::string_view second = "\r\n";
    ASSERT_EQ(parser.feed(second), ParseStatus::kComplete);
  }
  {
    HttpRequestParser parser;
    std::string_view first = head;
    ASSERT_EQ(parser.feed(first), ParseStatus::kNeedMoreData);
    std::string_view cr = "\r";
    ASSERT_EQ(parser.feed(cr), ParseStatus::kNeedMoreData);
    std::string_view lf = "\n";
    ASSERT_EQ(parser.feed(lf), ParseStatus::kComplete);
  }
}

// Boundary regression test for the header-count cap: exactly
// kMaxHeaderCount headers must be accepted.
TEST(HttpRequestParserTest, AcceptsExactlyMaxHeaderCount) {
  HttpRequestParser parser;
  std::string request = "GET / HTTP/1.1\r\n";
  for (std::size_t i = 0; i < HttpRequestParser::kMaxHeaderCount; ++i) {
    request += "X-H" + std::to_string(i) + ": v\r\n";
  }
  request += "\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().headers.size(), HttpRequestParser::kMaxHeaderCount);
}

// Regression test: the (kMaxHeaderCount + 1)th header must be rejected
// *before* being stored, not after. The buggy version incremented
// request_.headers (emplace_back) first and only checked size() >
// kMaxHeaderCount afterward, so it transiently held 101 headers.
TEST(HttpRequestParserTest, RejectsHeaderCountOneOverLimitWithoutStoringTheExtraHeader) {
  HttpRequestParser parser;
  std::string request = "GET / HTTP/1.1\r\n";
  for (std::size_t i = 0; i < HttpRequestParser::kMaxHeaderCount + 1; ++i) {
    request += "X-H" + std::to_string(i) + ": v\r\n";
  }
  request += "\r\n";
  std::string_view data = request;
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 431);
  EXPECT_EQ(parser.request().headers.size(), HttpRequestParser::kMaxHeaderCount)
      << "the header over the cap must never be stored, even transiently";
}

// RFC 7230 §3.3.3: duplicate Content-Length headers are a request-
// smuggling vector if taken as "first match wins" — a request whose second
// Content-Length disagrees with the first must be rejected outright, not
// parsed using only the first value while the rest of the declared-length
// mismatch is left to be misread as a pipelined request.
TEST(HttpRequestParserTest, RejectsDuplicateContentLengthWithDifferingValues) {
  HttpRequestParser parser;
  std::string_view data =
      "POST /echo HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 3\r\n\r\nhello";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

// Even identical duplicate values are rejected — the chosen policy is
// simpler and stricter than RFC 7230's "allow if all values match"
// carve-out, and no conforming client ever sends a duplicate at all.
TEST(HttpRequestParserTest, RejectsDuplicateContentLengthWithIdenticalValues) {
  HttpRequestParser parser;
  std::string_view data =
      "POST /echo HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 5\r\n\r\nhello";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

// Duplicate detection must be case-insensitive on the header name, same as
// every other header-name comparison in the parser.
TEST(HttpRequestParserTest, RejectsDuplicateContentLengthAcrossDifferingCase) {
  HttpRequestParser parser;
  std::string_view data =
      "POST /echo HTTP/1.1\r\nContent-Length: 5\r\ncontent-length: 5\r\n\r\nhello";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, RejectsMalformedHeaderLine) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.1\r\nNoColonHere\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kError);
  EXPECT_EQ(parser.error_response().status_code, 400);
}

TEST(HttpRequestParserTest, HeaderLookupIsCaseInsensitiveDuringParse) {
  HttpRequestParser parser;
  std::string_view data = "POST /echo HTTP/1.1\r\ncontent-length: 2\r\n\r\nhi";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().body, "hi");
}

TEST(HttpRequestParserTest, HeaderValueWhitespaceIsTrimmed) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.1\r\nX-Test:    padded value   \r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  auto value = find_header(parser.request().headers, "X-Test");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, "padded value");
}

TEST(HttpRequestParserTest, ResetAllowsReuseForNextRequest) {
  HttpRequestParser parser;
  std::string_view first = "GET /one HTTP/1.1\r\n\r\n";
  ASSERT_EQ(parser.feed(first), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().target, "/one");

  parser.reset();

  std::string_view second = "POST /two HTTP/1.1\r\nContent-Length: 1\r\n\r\nx";
  ASSERT_EQ(parser.feed(second), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().target, "/two");
  EXPECT_EQ(parser.request().body, "x");
  // Headers from the first request must not leak into the second.
  EXPECT_EQ(parser.request().headers.size(), 1u);
}

TEST(HttpRequestParserTest, ResetAllowsReuseAfterAnError) {
  HttpRequestParser parser;
  std::string_view bad = "DELETE / HTTP/1.1\r\n\r\n";
  ASSERT_EQ(parser.feed(bad), ParseStatus::kError);

  parser.reset();

  std::string_view good = "GET / HTTP/1.1\r\n\r\n";
  ASSERT_EQ(parser.feed(good), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().target, "/");
}

TEST(HttpRequestParserTest, ConnectionCloseReflectedOnRequest) {
  HttpRequestParser parser;
  std::string_view data = "GET / HTTP/1.1\r\nConnection: close\r\n\r\n";
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_FALSE(parser.request().keep_alive());
}

TEST(HttpRequestParserTest, PipeliningLeavesSecondRequestUnconsumed) {
  HttpRequestParser parser;
  std::string combined =
      "GET /first HTTP/1.1\r\n\r\n"
      "GET /second HTTP/1.1\r\n\r\n";
  std::string_view data = combined;

  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().target, "/first");
  EXPECT_EQ(data, "GET /second HTTP/1.1\r\n\r\n");

  parser.reset();
  ASSERT_EQ(parser.feed(data), ParseStatus::kComplete);
  EXPECT_EQ(parser.request().target, "/second");
  EXPECT_TRUE(data.empty());
}

}  // namespace
}  // namespace arcserve::protocol
