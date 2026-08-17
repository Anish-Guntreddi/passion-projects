#include "arcserve/app/cli.hpp"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace arcserve::app {
namespace {

std::optional<std::uint16_t> parse(std::initializer_list<const char*> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const char* arg : args) {
    argv.push_back(const_cast<char*>(arg));  // NOLINT(cppcoreguidelines-pro-type-const-cast)
  }
  return parse_port(static_cast<int>(argv.size()), argv.data());
}

TEST(ParsePortTest, DefaultsTo8080WhenNoArgumentGiven) {
  auto result = parse({"arcserve_server"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 8080);
}

TEST(ParsePortTest, ParsesOrdinaryValidPort) {
  auto result = parse({"arcserve_server", "9090"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 9090);
}

TEST(ParsePortTest, AcceptsZeroForEphemeralPort) {
  auto result = parse({"arcserve_server", "0"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 0);
}

TEST(ParsePortTest, AcceptsMaxValidPort65535) {
  auto result = parse({"arcserve_server", "65535"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 65535);
}

// Regression test: static_cast<uint16_t>(70000) used to silently wrap to
// 4464 instead of being rejected.
TEST(ParsePortTest, RejectsOutOfRangePortInsteadOfWrappingIt) {
  EXPECT_FALSE(parse({"arcserve_server", "70000"}).has_value());
}

// Regression test: static_cast<uint16_t>(-1) used to silently wrap to
// 65535 instead of being rejected.
TEST(ParsePortTest, RejectsNegativePortInsteadOfWrappingIt) {
  EXPECT_FALSE(parse({"arcserve_server", "-1"}).has_value());
}

TEST(ParsePortTest, RejectsNonNumericArgument) {
  EXPECT_FALSE(parse({"arcserve_server", "not-a-port"}).has_value());
}

TEST(ParsePortTest, RejectsEmptyArgument) {
  EXPECT_FALSE(parse({"arcserve_server", ""}).has_value());
}

TEST(ParsePortTest, RejectsTrailingGarbageAfterDigits) {
  EXPECT_FALSE(parse({"arcserve_server", "8080x"}).has_value());
}

// Regression test: std::stoi(argv[1]) used to throw std::out_of_range for
// input this large, which was never caught — an uncaught exception out of
// main() terminates the process instead of printing a usage error.
TEST(ParsePortTest, RejectsValueTooLargeToFitAnIntWithoutThrowing) {
  EXPECT_FALSE(parse({"arcserve_server", "999999999999999999999"}).has_value());
}

TEST(ParsePortTest, RejectsLeadingWhitespace) {
  EXPECT_FALSE(parse({"arcserve_server", " 8080"}).has_value());
}

TEST(ParsePortTest, RejectsLeadingPlusSign) {
  EXPECT_FALSE(parse({"arcserve_server", "+8080"}).has_value());
}

}  // namespace
}  // namespace arcserve::app
