#include "pebbledb/util/crc32c.hpp"

#include <string>

#include <gtest/gtest.h>

namespace pebbledb::util {
namespace {

TEST(Crc32c, EmptyInputIsZero) {
  EXPECT_EQ(Crc32c("", 0), 0u);
  EXPECT_EQ(Crc32c(std::string_view{}), 0u);
}

// Standard CRC-32C check value for the ASCII string "123456789", used to
// validate implementations against every other conformant CRC-32C
// implementation (e.g. RFC 3720/iSCSI, the "Catalogue of parametrised CRC
// algorithms" CRC-32C entry). This is a correctness fixture, not a
// benchmark number.
TEST(Crc32c, KnownCheckValue) {
  constexpr std::string_view kCheck = "123456789";
  EXPECT_EQ(Crc32c(kCheck), 0xE3069283u);
}

TEST(Crc32c, DifferentInputsProduceDifferentChecksums) {
  EXPECT_NE(Crc32c("abc"), Crc32c("abd"));
  EXPECT_NE(Crc32c("abc"), Crc32c("abcd"));
  EXPECT_NE(Crc32c(std::string(1, '\0')), Crc32c(std::string(2, '\0')));
}

TEST(Crc32c, DeterministicAcrossCalls) {
  std::string data = "the quick brown fox jumps over the lazy dog";
  EXPECT_EQ(Crc32c(data), Crc32c(data));
}

TEST(Crc32c, ExtendMatchesConcatenation) {
  std::string a = "hello, ";
  std::string b = "world!";
  std::uint32_t whole = Crc32c(a + b);

  std::uint32_t extended = Crc32c(a);
  extended = ExtendCrc32c(extended, b.data(), b.size());
  EXPECT_EQ(extended, whole);
}

TEST(Crc32c, ExtendFromZeroEqualsFreshChecksum) {
  std::string data = "some bytes";
  EXPECT_EQ(ExtendCrc32c(0, data.data(), data.size()), Crc32c(data));
}

TEST(Crc32c, ExtendAcrossManyChunksMatchesWhole) {
  std::string whole = "0123456789abcdefghijklmnopqrstuvwxyz";
  std::uint32_t crc = 0;
  for (char c : whole) {
    crc = ExtendCrc32c(crc, &c, 1);
  }
  EXPECT_EQ(crc, Crc32c(whole));
}

}  // namespace
}  // namespace pebbledb::util
