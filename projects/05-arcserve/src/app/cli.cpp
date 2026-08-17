#include "arcserve/app/cli.hpp"

#include <charconv>
#include <string_view>

namespace arcserve::app {

std::optional<std::uint16_t> parse_port(int argc, char** argv) noexcept {
  if (argc < 2) {
    return std::uint16_t{8080};
  }

  std::string_view text(argv[1]);
  if (text.empty()) {
    return std::nullopt;
  }

  // std::from_chars never throws and, unlike std::stoi, does not silently
  // accept leading whitespace or a leading '+' — it fails predictably
  // instead, and out-of-int-range input reports std::errc::result_out_of_range
  // rather than crashing the process via an uncaught std::out_of_range.
  int value = 0;
  const auto* begin = text.data();
  const auto* end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc() || ptr != end) {
    return std::nullopt;  // not a valid decimal integer, or trailing garbage
  }
  if (value < 0 || value > 65535) {
    return std::nullopt;  // outside the valid TCP port range — never wrap it
  }
  return static_cast<std::uint16_t>(value);
}

}  // namespace arcserve::app
