#include "arcserve/server/default_handlers.hpp"

#include <string>

namespace arcserve::server {

using protocol::HttpMethod;
using protocol::HttpRequest;
using protocol::HttpResponse;

HttpResponse default_route(const HttpRequest& request) {
  if (request.method == HttpMethod::kGet && request.target == "/") {
    HttpResponse response;
    response.status_code = 200;
    response.reason_phrase = "OK";
    response.headers.emplace_back("Content-Type", "text/plain");
    response.body = "ArcServe Phase 1 blocking server is alive.\n";
    return response;
  }

  if (request.method == HttpMethod::kPost && request.target == "/echo") {
    HttpResponse response;
    response.status_code = 200;
    response.reason_phrase = "OK";
    auto content_type = protocol::find_header(request.headers, "Content-Type");
    response.headers.emplace_back("Content-Type",
                                   content_type ? std::string(*content_type) : "text/plain");
    response.body = request.body;
    return response;
  }

  return protocol::make_error_response(404, "Not Found", "no such route\n", /*keep_alive=*/true);
}

}  // namespace arcserve::server
