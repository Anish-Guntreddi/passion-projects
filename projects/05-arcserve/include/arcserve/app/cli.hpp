#pragma once

#include <cstdint>
#include <optional>

namespace arcserve::app {

// Parses the server's single optional CLI argument (argv[1]) as a TCP port
// number.
//
//   - argc < 2 (no argument given): returns 8080, the default port.
//   - argv[1] present: must be a decimal integer with no sign, no leading
//     '+', no surrounding whitespace, and no trailing garbage, and its
//     value must fit in [0, 65535] (0 = kernel-assigned ephemeral port,
//     matching BlockingHttpServer::Config::port). Anything else — a
//     non-numeric token, a negative number, a number above 65535, or a
//     number too large to parse at all — returns std::nullopt rather than
//     silently truncating/wrapping via a narrowing cast (e.g. 70000 must
//     never become 4464) or throwing an uncaught exception out of main().
//     Callers should print a usage error and exit nonzero on nullopt.
[[nodiscard]] std::optional<std::uint16_t> parse_port(int argc, char** argv) noexcept;

}  // namespace arcserve::app
