#include "arcserve/protocol/http_message.hpp"

#include <algorithm>
#include <cctype>

namespace arcserve::protocol {

bool ascii_case_insensitive_equals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  return std::equal(a.begin(), a.end(), b.begin(), [](unsigned char lhs, unsigned char rhs) {
    return std::tolower(lhs) == std::tolower(rhs);
  });
}

std::string_view method_to_string(HttpMethod method) noexcept {
  switch (method) {
    case HttpMethod::kGet:
      return "GET";
    case HttpMethod::kPost:
      return "POST";
  }
  return "GET";  // unreachable for a valid HttpMethod; keeps the function total
}

std::optional<std::string_view> find_header(const HeaderList& headers,
                                             std::string_view name) noexcept {
  for (const auto& [key, value] : headers) {
    if (ascii_case_insensitive_equals(key, name)) {
      return std::string_view(value);
    }
  }
  return std::nullopt;
}

bool HttpRequest::keep_alive() const noexcept {
  if (auto connection = find_header(headers, "Connection")) {
    return !ascii_case_insensitive_equals(*connection, "close");
  }
  return true;  // HTTP/1.1 default (RFC 9112 §9.3)
}

std::string HttpResponse::serialize() const {
  std::string out;
  out.reserve(128 + body.size());

  out += "HTTP/1.1 ";
  out += std::to_string(status_code);
  out += ' ';
  out += reason_phrase;
  out += "\r\n";

  for (const auto& [key, value] : headers) {
    out += key;
    out += ": ";
    out += value;
    out += "\r\n";
  }

  out += "Content-Length: ";
  out += std::to_string(body.size());
  out += "\r\n";

  out += "Connection: ";
  out += (keep_alive ? "keep-alive" : "close");
  out += "\r\n\r\n";

  out += body;
  return out;
}

HttpResponse make_error_response(int status_code, std::string_view reason_phrase,
                                  std::string_view body, bool keep_alive) {
  HttpResponse response;
  response.status_code = status_code;
  response.reason_phrase = std::string(reason_phrase);
  response.body = std::string(body);
  response.keep_alive = keep_alive;
  response.headers.emplace_back("Content-Type", "text/plain");
  return response;
}

}  // namespace arcserve::protocol
