#include "arcserve/protocol/http_message.hpp"

#include <string>

#include <gtest/gtest.h>

namespace arcserve::protocol {
namespace {

TEST(HttpMessageTest, FindHeaderIsCaseInsensitive) {
  HeaderList headers{{"Content-Type", "text/plain"}, {"X-Test", "1"}};
  auto found = find_header(headers, "content-type");
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, "text/plain");
}

TEST(HttpMessageTest, FindHeaderReturnsNulloptWhenMissing) {
  HeaderList headers{{"X-Test", "1"}};
  EXPECT_FALSE(find_header(headers, "Content-Type").has_value());
}

TEST(HttpMessageTest, FindHeaderReturnsFirstMatchOnDuplicates) {
  HeaderList headers{{"X-Test", "first"}, {"X-Test", "second"}};
  auto found = find_header(headers, "X-Test");
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, "first");
}

TEST(HttpMessageTest, KeepAliveDefaultsTrueForHttp11) {
  HttpRequest request;
  EXPECT_TRUE(request.keep_alive());
}

TEST(HttpMessageTest, KeepAliveFalseWhenConnectionCloseHeaderPresent) {
  HttpRequest request;
  request.headers.emplace_back("Connection", "close");
  EXPECT_FALSE(request.keep_alive());
}

TEST(HttpMessageTest, KeepAliveFalseIsCaseInsensitive) {
  HttpRequest request;
  request.headers.emplace_back("connection", "Close");
  EXPECT_FALSE(request.keep_alive());
}

TEST(HttpMessageTest, KeepAliveTrueWhenConnectionKeepAliveHeaderPresent) {
  HttpRequest request;
  request.headers.emplace_back("Connection", "keep-alive");
  EXPECT_TRUE(request.keep_alive());
}

TEST(HttpMessageTest, SerializeProducesWellFormedResponse) {
  HttpResponse response;
  response.status_code = 200;
  response.reason_phrase = "OK";
  response.headers.emplace_back("Content-Type", "text/plain");
  response.body = "hello";
  response.keep_alive = true;

  std::string wire = response.serialize();

  EXPECT_EQ(wire.substr(0, 17), "HTTP/1.1 200 OK\r\n");
  EXPECT_NE(wire.find("Content-Type: text/plain\r\n"), std::string::npos);
  EXPECT_NE(wire.find("Content-Length: 5\r\n"), std::string::npos);
  EXPECT_NE(wire.find("Connection: keep-alive\r\n"), std::string::npos);
  EXPECT_TRUE(wire.ends_with("\r\n\r\nhello"));
}

TEST(HttpMessageTest, SerializeReflectsConnectionClose) {
  HttpResponse response;
  response.keep_alive = false;
  std::string wire = response.serialize();
  EXPECT_NE(wire.find("Connection: close\r\n"), std::string::npos);
}

TEST(HttpMessageTest, SerializeComputesContentLengthFromBodySize) {
  HttpResponse response;
  response.body = std::string(1234, 'x');
  std::string wire = response.serialize();
  EXPECT_NE(wire.find("Content-Length: 1234\r\n"), std::string::npos);
}

TEST(HttpMessageTest, MakeErrorResponseSetsAllFields) {
  HttpResponse response = make_error_response(400, "Bad Request", "bad\n", /*keep_alive=*/false);
  EXPECT_EQ(response.status_code, 400);
  EXPECT_EQ(response.reason_phrase, "Bad Request");
  EXPECT_EQ(response.body, "bad\n");
  EXPECT_FALSE(response.keep_alive);
}

TEST(HttpMessageTest, MethodToStringRoundTrip) {
  EXPECT_EQ(method_to_string(HttpMethod::kGet), "GET");
  EXPECT_EQ(method_to_string(HttpMethod::kPost), "POST");
}

}  // namespace
}  // namespace arcserve::protocol
