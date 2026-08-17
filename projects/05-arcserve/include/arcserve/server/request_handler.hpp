#pragma once

#include <functional>

#include "arcserve/protocol/http_message.hpp"

namespace arcserve::server {

// Given a fully-parsed request, produce the response to send. Shared by
// every HTTP server variant in this codebase — BlockingHttpServer (Phase
// 1) and NonblockingHttpServer (Phase 3) both accept exactly this type, so
// a handler written once (e.g. default_route) works unmodified against
// either. Invoked synchronously on whichever thread is driving the
// connection at the time (the single serving thread in Phase 1; the single
// reactor thread in Phase 3) — there is no worker pool yet (that's Phase
// 5), so a slow handler still blocks that one thread's other work.
using RequestHandler = std::function<protocol::HttpResponse(const protocol::HttpRequest&)>;

}  // namespace arcserve::server
