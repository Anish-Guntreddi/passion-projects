# Protocol scope: ArcServe's HTTP/1.1 subset

This document is the exact scope referenced by
[`docs/decisions/0002-protocol-scope-http-1.1-subset.md`](decisions/0002-protocol-scope-http-1.1-subset.md)
(spec open decision D1). It is the contract `HttpRequestParser`
(`include/arcserve/protocol/http_parser.hpp`) implements and the integration
tests in `tests/integration/` verify against. Anything not listed here is
out of scope for the MVP and is rejected predictably rather than
best-effort-parsed.

## Request line

```
METHOD SP target SP HTTP-version CRLF
```

- **METHOD**: exactly `GET` or `POST` (case-sensitive, per RFC 9110 — HTTP
  method tokens are case-sensitive; `get` is not `GET`). Anything else →
  **501 Not Implemented**.
- **target**: origin-form only, i.e. it must start with `/`
  (`http://host/path` absolute-form, `*` asterisk-form, and
  `host:port` authority-form are all rejected). Query strings are treated
  as opaque bytes within the target (no decomposition). Anything else →
  **400 Bad Request**.
- **HTTP-version**: must be exactly `HTTP/1.1`. `HTTP/1.0`, `HTTP/2`, and
  malformed version tokens → **505 HTTP Version Not Supported**.
- Total request-line length (method + target + version, before the
  terminating CRLF) is capped at `HttpRequestParser::kMaxRequestLineLength`
  (8192 bytes) → **400 Bad Request** if exceeded.

## Headers

- Standard `Name: value` lines, terminated by CRLF, ending with a single
  empty CRLF line.
- Header names are matched case-insensitively (RFC 9110 §5.1) for internal
  lookups (`Content-Length`, `Connection`); names and values are otherwise
  stored and echoed verbatim (no canonicalization of casing on the wire).
- Optional leading/trailing whitespace (space/tab) around a header value is
  trimmed (RFC 9110 §5.5, "optional whitespace").
- A line with no `:` — or a `:` as the very first character (empty name) —
  is malformed → **400 Bad Request**.
- Limits, enforced while parsing (not after buffering everything):
  - Total header bytes ≤ `kMaxHeaderBytes` (8192), counting each header
    line's terminating CRLF (unlike the request-line cap above, which is
    measured *before* its CRLF) → **431 Request Header Fields Too
    Large**.
  - Header count ≤ `kMaxHeaderCount` (100) → **431 Request Header Fields
    Too Large**. Enforced before the 101st header is ever stored, not
    after.
- **Multiple `Content-Length` headers are rejected**, even if every
  duplicate carries the identical value → **400 Bad Request** (RFC 7230
  §3.3.3: duplicate `Content-Length` fields are a request-smuggling vector
  if a value other than the first is later treated as authoritative
  anywhere in the request's path; no conforming client ever sends more
  than one).
- **No chunked transfer-encoding.** Body framing is Content-Length only —
  this is the single biggest scope cut relative to full HTTP/1.1, and is
  what keeps the parser small enough to stay hand-written and
  interview-explainable (see ADR 0003). A `Transfer-Encoding` header is
  accepted but ignored in Phase 1 (chunked bodies are simply not supported;
  a client that sends one without Content-Length will hit the "POST without
  Content-Length" case below).

## Body

- Framed exclusively by `Content-Length` (decimal, no leading `+`/`-`, no
  whitespace inside the digits — validated via `std::from_chars`).
  - Invalid value (non-numeric, trailing garbage) → **400 Bad Request**.
  - Value `> kMaxBodyBytes` (1 MiB) → **413 Payload Too Large**, raised
    immediately when the header is parsed, *before* any body bytes are
    read off the socket — an oversized declared length is rejected without
    ever allocating or reading the oversized payload.
- **POST requires `Content-Length`.** A POST with no such header →
  **411 Length Required**.
- **GET does not require it.** If absent, the body is treated as
  zero-length. If a GET does carry `Content-Length`, it is honored (GET
  with a body is unusual but not forbidden by HTTP semantics; the MVP
  doesn't special-case it away).

## Connection handling

- HTTP/1.1 connections default to persistent (`keep-alive`) per RFC 9112
  §9.3. A `Connection: close` request header (case-insensitive match on
  the value) closes the connection after that response.
- Every response ArcServe sends carries an explicit `Connection` header
  (`keep-alive` or `close`) and an explicit `Content-Length` — both are
  computed and appended by `HttpResponse::serialize()`, never left to be
  inferred by the client.
- All parser-detected errors (400/411/413/431/501/505) close the
  connection after the error response is sent — a client whose request the
  server couldn't parse doesn't get another attempt on the same connection;
  it reconnects.
- A single connection is bounded to
  `BlockingHttpServer::Config::max_keep_alive_requests` (default 1000)
  requests, after which the server closes it regardless of the
  `Connection` header — an unbounded keep-alive loop on one connection
  would be a liveness bug for every other client waiting in the accept
  backlog (Phase 1's server handles one connection at a time; see
  `docs/architecture.md`).

## Status codes this MVP emits

| Code | Reason phrase | Trigger |
|---|---|---|
| 200 | OK | Successful GET `/` or POST `/echo` |
| 400 | Bad Request | Malformed request line, malformed header line, non-origin-form target, invalid Content-Length, duplicate Content-Length |
| 404 | Not Found | Any route other than `GET /` or `POST /echo` |
| 411 | Length Required | POST without Content-Length |
| 413 | Payload Too Large | Content-Length exceeds `kMaxBodyBytes` |
| 431 | Request Header Fields Too Large | Header bytes or count exceed their caps |
| 501 | Not Implemented | Method other than GET/POST |
| 505 | HTTP Version Not Supported | Version other than HTTP/1.1 |

## Explicitly out of scope for the MVP

Matches the spec's non-goals (§1.4): chunked transfer-encoding, any method
beyond GET/POST, HTTP/1.0 and HTTP/2+, TLS, absolute-form/authority-form/
asterisk-form request targets, trailers, `Expect: 100-continue`, and range
requests. Revisiting any of these is a new ADR, not a silent scope creep.
