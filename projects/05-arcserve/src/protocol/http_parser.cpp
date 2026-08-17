#include "arcserve/protocol/http_parser.hpp"

#include <algorithm>
#include <charconv>

namespace arcserve::protocol {

ParseStatus HttpRequestParser::feed(std::string_view& data) {
  for (;;) {
    StepResult result;
    switch (state_) {
      case State::kRequestLine:
        result = parse_request_line(data);
        break;
      case State::kHeaders:
        result = parse_headers(data);
        break;
      case State::kBody:
        result = parse_body(data);
        break;
      case State::kDone:
        // A caller that forgets reset() after kComplete/kError and feeds
        // again lands here; report completion rather than silently eating
        // bytes that were never inspected.
        return ParseStatus::kComplete;
    }

    switch (result) {
      case StepResult::kNeedMoreData:
        return ParseStatus::kNeedMoreData;
      case StepResult::kFailed:
        return ParseStatus::kError;
      case StepResult::kAdvanced:
        if (state_ == State::kDone) {
          return ParseStatus::kComplete;
        }
        continue;  // state_ moved forward (e.g. RequestLine -> Headers);
                   // re-dispatch on the remaining `data` in this same call.
    }
  }
}

void HttpRequestParser::reset() {
  state_ = State::kRequestLine;
  line_buffer_.clear();
  header_bytes_seen_ = 0;
  content_length_.reset();
  request_ = HttpRequest{};
  error_response_ = HttpResponse{};
}

HttpRequestParser::StepResult HttpRequestParser::fail_step(int status_code,
                                                             std::string_view reason_phrase,
                                                             std::string_view body) {
  error_response_ = make_error_response(status_code, reason_phrase, body, /*keep_alive=*/false);
  state_ = State::kDone;  // Terminal: caller must reset() before reusing this parser.
  return StepResult::kFailed;
}

// --- Request line ---------------------------------------------------------
//
// Invariant maintained across calls: line_buffer_ never contains an
// embedded "\r\n" (we transition out of the line-accumulation state the
// instant one is found), so a terminator can only appear (a) fully inside
// the newly-arrived `data`, or (b) split across the line_buffer_/data
// boundary as a trailing '\r' + a leading '\n'. Both cases are handled
// explicitly below; no other case is possible.
HttpRequestParser::StepResult HttpRequestParser::parse_request_line(std::string_view& data) {
  if (!line_buffer_.empty() && line_buffer_.back() == '\r' && !data.empty() &&
      data.front() == '\n') {
    data.remove_prefix(1);
    std::string_view line(line_buffer_.data(), line_buffer_.size() - 1);
    StepResult result = finish_request_line(line);
    line_buffer_.clear();
    return result;
  }

  auto crlf = data.find("\r\n");
  if (crlf != std::string_view::npos) {
    if (line_buffer_.size() + crlf > kMaxRequestLineLength) {
      return fail_step(400, "Bad Request", "request line exceeds limit\n");
    }
    line_buffer_.append(data.data(), crlf);
    data.remove_prefix(crlf + 2);
    std::string_view line(line_buffer_);
    StepResult result = finish_request_line(line);
    line_buffer_.clear();
    return result;
  }

  // No terminator yet in this chunk (and the boundary case above didn't
  // apply). Buffer everything we have, bounded, and wait for more.
  //
  // Exception: if `data` is exactly a single trailing '\r', it might be the
  // first half of the terminating CRLF, split across TCP segments (a lone
  // '\r' now, its pairing '\n' next call) — the boundary-consuming branch
  // at the top of this function excludes a terminator's own '\r' from the
  // line's actual content once its '\n' arrives, so that '\r' must not be
  // charged against the cap here either. Without this, a request line whose
  // real content lands exactly at kMaxRequestLineLength is wrongly rejected
  // purely because its CRLF happened to arrive split rather than whole
  // (mirrors parse_headers()'s identical `pending_could_still_be_terminator`
  // guard for the header-block terminator). If the next byte turns out not
  // to be '\n', that '\r' really was content instead, and this same check —
  // now with it already included in line_buffer_ — catches any true
  // overflow on the very next call.
  bool pending_could_be_terminator_start = data.size() == 1 && data.front() == '\r';
  if (!pending_could_be_terminator_start &&
      line_buffer_.size() + data.size() > kMaxRequestLineLength) {
    return fail_step(400, "Bad Request", "request line exceeds limit\n");
  }
  line_buffer_.append(data);
  data.remove_prefix(data.size());
  return StepResult::kNeedMoreData;
}

HttpRequestParser::StepResult HttpRequestParser::finish_request_line(std::string_view line) {
  auto first_space = line.find(' ');
  if (first_space == std::string_view::npos) {
    return fail_step(400, "Bad Request", "malformed request line\n");
  }
  std::string_view method_str = line.substr(0, first_space);
  std::string_view rest = line.substr(first_space + 1);

  auto second_space = rest.find(' ');
  if (second_space == std::string_view::npos) {
    return fail_step(400, "Bad Request", "malformed request line\n");
  }
  std::string_view target = rest.substr(0, second_space);
  std::string_view version = rest.substr(second_space + 1);

  if (target.empty() || version.empty() || target.find(' ') != std::string_view::npos) {
    return fail_step(400, "Bad Request", "malformed request line\n");
  }

  if (method_str == "GET") {
    request_.method = HttpMethod::kGet;
  } else if (method_str == "POST") {
    request_.method = HttpMethod::kPost;
  } else {
    return fail_step(501, "Not Implemented", "unsupported method\n");
  }

  if (target.front() != '/') {
    return fail_step(400, "Bad Request", "request target must be origin-form\n");
  }
  request_.target.assign(target);

  if (version != "HTTP/1.1") {
    return fail_step(505, "HTTP Version Not Supported", "only HTTP/1.1 is supported\n");
  }
  request_.version.assign(version);

  state_ = State::kHeaders;
  header_bytes_seen_ = 0;
  return StepResult::kAdvanced;
}

// --- Headers ----------------------------------------------------------------
//
// Loops internally over as many complete header lines as `data` contains
// (a full header block commonly arrives in one recv()); returns to feed()
// once it either needs more bytes or leaves the Headers state entirely
// (empty line -> finish_headers() moves state_ to kBody or kDone).
HttpRequestParser::StepResult HttpRequestParser::parse_headers(std::string_view& data) {
  for (;;) {
    if (!line_buffer_.empty() && line_buffer_.back() == '\r' && !data.empty() &&
        data.front() == '\n') {
      data.remove_prefix(1);
      std::string_view line(line_buffer_.data(), line_buffer_.size() - 1);
      StepResult result = consume_header_line(line);
      line_buffer_.clear();
      if (result != StepResult::kAdvanced || state_ != State::kHeaders) {
        return result;
      }
      continue;
    }

    auto crlf = data.find("\r\n");
    if (crlf == std::string_view::npos) {
      // The pending (still-unterminated) line so far might still turn out
      // to be the empty line that terminates the header block — that can
      // only be true if line_buffer_ is empty and `data` holds nothing but
      // a leading '\r' still waiting on its '\n' (anything else already
      // proves this line has real content). The terminator itself is not a
      // header and must never be charged against the cap (consume_header_line()
      // grants it the same exemption via its own `line.empty()` check,
      // which runs before any size accounting) — otherwise a header block
      // whose real header bytes land exactly at kMaxHeaderBytes gets
      // rejected here before the terminator that would complete it, and
      // legitimately-sized traffic starts failing depending on chunk
      // boundaries alone.
      bool pending_could_still_be_terminator =
          line_buffer_.empty() && (data.empty() || (data.size() == 1 && data.front() == '\r'));
      if (!pending_could_still_be_terminator &&
          header_bytes_seen_ + line_buffer_.size() + data.size() > kMaxHeaderBytes) {
        return fail_step(431, "Request Header Fields Too Large", "headers exceed limit\n");
      }
      line_buffer_.append(data);
      data.remove_prefix(data.size());
      return StepResult::kNeedMoreData;
    }

    // +2 for this line's own CRLF: header_bytes_seen_ must track true wire
    // bytes (docs/protocol-scope.md: "Total header bytes ≤ kMaxHeaderBytes"
    // — no CRLF exclusion, unlike the request line's documented cap). This
    // is an early-out; consume_header_line() below re-checks authoritatively
    // (it's also reached via the line_buffer_/data split-boundary path a
    // few lines up, which has no size check of its own).
    //
    // Exception: the empty line that terminates the header block
    // (line_buffer_ empty and zero content before this CRLF) is not itself
    // a header and must never be charged against the cap here either —
    // same reasoning as the no-CRLF-yet branch just above.
    bool is_header_block_terminator = line_buffer_.empty() && crlf == 0;
    if (!is_header_block_terminator &&
        header_bytes_seen_ + line_buffer_.size() + crlf + 2 > kMaxHeaderBytes) {
      return fail_step(431, "Request Header Fields Too Large", "headers exceed limit\n");
    }
    line_buffer_.append(data.data(), crlf);
    data.remove_prefix(crlf + 2);
    std::string_view line(line_buffer_);
    StepResult result = consume_header_line(line);
    line_buffer_.clear();
    if (result != StepResult::kAdvanced || state_ != State::kHeaders) {
      return result;
    }
    // else: a normal header line was consumed and we're still parsing
    // headers — loop for the next one using whatever remains of `data`.
  }
}

HttpRequestParser::StepResult HttpRequestParser::consume_header_line(std::string_view line) {
  if (line.empty()) {
    return finish_headers();
  }

  // RFC 9110 §5.5 forbids CR, LF, and NUL inside a field value; this
  // parser's line scanner (parse_headers(), above) only recognizes the
  // literal 2-byte "\r\n" sequence as a line terminator, so a bare CR or LF
  // that isn't part of an actual CRLF pair survives into `line` unrejected
  // instead. Left unchecked, a request like
  // "Content-Type: text/plain\nSet-Cookie: admin=true\r\n" parses as one
  // header whose value carries an embedded LF, and any handler that
  // reflects that value back into a response header (e.g. default_route's
  // /echo route) hands the client a way to inject an extra header/status
  // line into ArcServe's own response (CWE-113, HTTP response splitting).
  // Reject outright, matching FR3's "malformed input is rejected
  // predictably".
  if (line.find('\r') != std::string_view::npos || line.find('\n') != std::string_view::npos) {
    return fail_step(400, "Bad Request", "malformed header line\n");
  }

  auto colon = line.find(':');
  if (colon == std::string_view::npos || colon == 0) {
    return fail_step(400, "Bad Request", "malformed header line\n");
  }

  std::string_view name = line.substr(0, colon);
  std::string_view value = line.substr(colon + 1);
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1);
  }

  if (name.find(' ') != std::string_view::npos || name.find('\t') != std::string_view::npos) {
    return fail_step(400, "Bad Request", "malformed header name\n");
  }

  // Authoritative byte-cap check: +2 accounts for this line's own CRLF, so
  // header_bytes_seen_ — once updated below — never itself exceeds
  // kMaxHeaderBytes. Checked here (not just in parse_headers()'s early-out)
  // because this function is also reached directly from the
  // line_buffer_/data split-boundary path in parse_headers(), which has no
  // size check of its own before calling in.
  if (header_bytes_seen_ + line.size() + 2 > kMaxHeaderBytes) {
    return fail_step(431, "Request Header Fields Too Large", "headers exceed limit\n");
  }
  header_bytes_seen_ += line.size() + 2;

  // RFC 7230 §3.3.3: a request with multiple Content-Length header fields
  // is ambiguous (and a classic request-smuggling vector if a downstream
  // server picks a different one of the duplicates than we do) — reject
  // outright rather than taking find_header()'s first match and silently
  // treating the rest of a mismatched declared length as pipelined data.
  // Simpler and stricter than RFC 7230's "allow if all values are
  // identical" carve-out, and no conforming client ever needs it.
  if (ascii_case_insensitive_equals(name, "Content-Length") &&
      find_header(request_.headers, "Content-Length").has_value()) {
    return fail_step(400, "Bad Request", "duplicate Content-Length header\n");
  }

  // Reject before storing the (kMaxHeaderCount + 1)th header rather than
  // after — the documented cap is "header count ≤ kMaxHeaderCount", so the
  // parser must never itself hold more than that many headers, even
  // transiently.
  if (request_.headers.size() >= kMaxHeaderCount) {
    return fail_step(431, "Request Header Fields Too Large", "too many headers\n");
  }
  request_.headers.emplace_back(std::string(name), std::string(value));

  return StepResult::kAdvanced;  // still in kHeaders; caller loops for the next line
}

HttpRequestParser::StepResult HttpRequestParser::finish_headers() {
  if (auto content_length_header = find_header(request_.headers, "Content-Length")) {
    std::size_t value = 0;
    const char* begin = content_length_header->data();
    const char* end = begin + content_length_header->size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc() || ptr != end) {
      return fail_step(400, "Bad Request", "invalid Content-Length\n");
    }
    if (value > kMaxBodyBytes) {
      return fail_step(413, "Payload Too Large", "body exceeds limit\n");
    }
    content_length_ = value;
  } else if (request_.method == HttpMethod::kPost) {
    return fail_step(411, "Length Required", "POST requires Content-Length\n");
  } else {
    content_length_ = 0;
  }

  request_.body.clear();
  if (*content_length_ == 0) {
    state_ = State::kDone;
  } else {
    request_.body.reserve(*content_length_);
    state_ = State::kBody;
  }
  return StepResult::kAdvanced;
}

// --- Body ---------------------------------------------------------------
HttpRequestParser::StepResult HttpRequestParser::parse_body(std::string_view& data) {
  std::size_t remaining = *content_length_ - request_.body.size();
  std::size_t take = std::min(remaining, data.size());
  request_.body.append(data.data(), take);
  data.remove_prefix(take);

  if (request_.body.size() < *content_length_) {
    return StepResult::kNeedMoreData;
  }
  state_ = State::kDone;
  return StepResult::kAdvanced;
}

}  // namespace arcserve::protocol
