#pragma once

#include "arcserve/protocol/http_message.hpp"

namespace arcserve::server {

// The Phase 1 reference handler: a tiny fixed route table shared by the
// production binary (src/app/main.cpp) and the integration tests, so what
// gets tested is exactly what ships.
//
//   GET  /      -> 200, fixed text body
//   POST /echo  -> 200, request body/Content-Type mirrored back
//   anything else -> 404
[[nodiscard]] protocol::HttpResponse default_route(const protocol::HttpRequest& request);

}  // namespace arcserve::server
