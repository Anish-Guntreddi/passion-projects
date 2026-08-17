#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>

#include "arcserve/app/cli.hpp"
#include "arcserve/server/blocking_server.hpp"
#include "arcserve/server/default_handlers.hpp"

namespace {

// A pointer-to-server global, set once before signals are ever unblocked
// and never reseated afterward, so the handler below only ever performs
// plain lock-free atomic loads/stores — the one category of standard
// library operation the C++ standard explicitly permits inside a signal
// handler ([support.signal]). The static_asserts make that permission
// load-bearing rather than assumed: if either type were not
// always-lock-free on some future target, this would fail to compile
// instead of silently becoming signal-unsafe.
std::atomic<arcserve::server::BlockingHttpServer*> g_server{nullptr};
static_assert(std::atomic<arcserve::server::BlockingHttpServer*>::is_always_lock_free);

void handle_signal(int /*signal_number*/) {
  if (arcserve::server::BlockingHttpServer* server = g_server.load()) {
    server->stop();  // BlockingHttpServer::stop() is itself just one
                      // relaxed store to a std::atomic<bool> — see its
                      // static_assert in blocking_server.hpp.
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::uint16_t> port = arcserve::app::parse_port(argc, argv);
  if (!port.has_value()) {
    std::cerr << "usage: " << (argc > 0 ? argv[0] : "arcserve_server")
              << " [port]\n"
                 "  port must be a decimal integer in [0, 65535] "
                 "(0 = kernel-assigned ephemeral port); default 8080\n";
    return EXIT_FAILURE;
  }

  arcserve::server::BlockingHttpServer::Config config;
  config.port = *port;

  arcserve::server::BlockingHttpServer server(config, arcserve::server::default_route);
  g_server.store(&server);
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  std::cout << "arcserve_server listening on 127.0.0.1:" << server.port()
            << " (blocking, Phase 1 — single connection at a time)\n";
  server.serve();
  std::cout << "arcserve_server stopped\n";
  return EXIT_SUCCESS;
}
