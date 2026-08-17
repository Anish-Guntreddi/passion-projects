#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arcserve::protocol {

// The two methods ArcServe's HTTP/1.1 subset understands. Anything else in
// a request line is a parse error (501), not a value of this enum — see
// docs/protocol-scope.md.
enum class HttpMethod { kGet, kPost };

[[nodiscard]] std::string_view method_to_string(HttpMethod method) noexcept;

// Ordered, case-insensitively-matched header list. A vector of pairs
// (rather than a map) because the protocol scope caps header count low
// enough (HttpRequestParser::kMaxHeaderCount) that O(n) lookup cost is
// irrelevant, and preserving wire order keeps parsed/serialized output
// predictable to assert against in tests.
using HeaderList = std::vector<std::pair<std::string, std::string>>;

// Case-insensitive header lookup (RFC 9110 §5.1: field names are
// case-insensitive). Returns the first match, if any.
[[nodiscard]] std::optional<std::string_view> find_header(const HeaderList& headers,
                                                            std::string_view name) noexcept;

// Case-insensitive ASCII equality, shared by header *name* matching (RFC
// 9110 §5.1 — field names are case-insensitive) and header *value* token
// matching (e.g. "Connection: close" vs. "Connection: Close"). Exposed
// (rather than kept private to this translation unit) so callers that need
// to reason about header names before they're in a HeaderList — e.g.
// HttpRequestParser detecting a duplicate Content-Length header while
// still consuming one line at a time — reuse the exact same comparison
// find_header and HttpRequest::keep_alive() use, instead of a second
// hand-rolled one.
[[nodiscard]] bool ascii_case_insensitive_equals(std::string_view a, std::string_view b) noexcept;

struct HttpRequest {
  HttpMethod method = HttpMethod::kGet;
  std::string target;   // origin-form request target, e.g. "/echo"
  std::string version;  // always "HTTP/1.1" on a successfully-parsed request
  HeaderList headers;
  std::string body;

  // False only when a "Connection: close" header is present; HTTP/1.1
  // connections default to persistent (RFC 9112 §9.3).
  [[nodiscard]] bool keep_alive() const noexcept;
};

struct HttpResponse {
  int status_code = 200;
  std::string reason_phrase = "OK";
  HeaderList headers;
  std::string body;
  bool keep_alive = true;

  // Renders the full HTTP/1.1 response. Content-Length is computed from
  // body.size() and Connection is derived from `keep_alive`; both are
  // appended automatically — callers must not add their own Content-Length
  // or Connection entries to `headers` (doing so would duplicate them on
  // the wire).
  [[nodiscard]] std::string serialize() const;
};

// Builds a canned error/status response. Used both by HttpRequestParser
// (protocol errors) and by application handlers (e.g. 404 for unknown
// routes) so status text stays consistent across the codebase.
[[nodiscard]] HttpResponse make_error_response(int status_code, std::string_view reason_phrase,
                                                std::string_view body, bool keep_alive);

}  // namespace arcserve::protocol
