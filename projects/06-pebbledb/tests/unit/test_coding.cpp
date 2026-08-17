#include "pebbledb/util/coding.hpp"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace pebbledb::util {
namespace {

TEST(Coding, Fixed32RoundTripBoundaryValues) {
  const std::uint32_t values[] = {0u, 1u, 255u, 256u, 0x12345678u,
                                   std::numeric_limits<std::uint32_t>::max()};
  for (std::uint32_t v : values) {
    std::string buf;
    PutFixed32(&buf, v);
    ASSERT_EQ(buf.size(), 4u);
    EXPECT_EQ(DecodeFixed32(buf.data()), v);
  }
}

TEST(Coding, Fixed64RoundTripBoundaryValues) {
  const std::uint64_t values[] = {0ull, 1ull, 0x1122334455667788ull,
                                   std::numeric_limits<std::uint64_t>::max()};
  for (std::uint64_t v : values) {
    std::string buf;
    PutFixed64(&buf, v);
    ASSERT_EQ(buf.size(), 8u);
    EXPECT_EQ(DecodeFixed64(buf.data()), v);
  }
}

TEST(Coding, Fixed32IsLittleEndianOnDisk) {
  std::string buf;
  PutFixed32(&buf, 0x01020304u);
  ASSERT_EQ(buf.size(), 4u);
  // Least-significant byte first.
  EXPECT_EQ(static_cast<unsigned char>(buf[0]), 0x04u);
  EXPECT_EQ(static_cast<unsigned char>(buf[1]), 0x03u);
  EXPECT_EQ(static_cast<unsigned char>(buf[2]), 0x02u);
  EXPECT_EQ(static_cast<unsigned char>(buf[3]), 0x01u);
}

TEST(Coding, Fixed64IsLittleEndianOnDisk) {
  std::string buf;
  PutFixed64(&buf, 0x0102030405060708ull);
  ASSERT_EQ(buf.size(), 8u);
  EXPECT_EQ(static_cast<unsigned char>(buf[0]), 0x08u);
  EXPECT_EQ(static_cast<unsigned char>(buf[7]), 0x01u);
}

TEST(Coding, PutAppendsRatherThanOverwrites) {
  std::string buf = "prefix:";
  PutFixed32(&buf, 42u);
  EXPECT_EQ(buf.size(), 7u + 4u);
  EXPECT_EQ(buf.substr(0, 7), "prefix:");
  EXPECT_EQ(DecodeFixed32(buf.data() + 7), 42u);
}

}  // namespace
}  // namespace pebbledb::util
