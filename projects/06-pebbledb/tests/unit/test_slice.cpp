#include "pebbledb/slice.hpp"

#include <gtest/gtest.h>

namespace pebbledb {
namespace {

// Slice (== std::string_view) must be binary-safe: length-delimited, not
// NUL-terminated. This is the property the rest of the codebase (WAL
// encode/decode, DB::Put/Get) relies on for "keys/values are binary-safe
// bytes" (spec §1.2). See slice.hpp for the full ownership-rule
// documentation this test file backs up.

TEST(Slice, SizeReflectsEmbeddedNulBytes) {
  std::string raw("ab\0cd", 5);
  Slice s = raw;
  EXPECT_EQ(s.size(), 5u);
  EXPECT_NE(s.size(), std::string_view(raw.c_str()).size());  // NUL-terminated view would be 2
}

TEST(Slice, CoversFullByteRangeIncludingHighBytes) {
  std::string raw;
  for (int b = 0; b < 256; ++b) {
    raw.push_back(static_cast<char>(static_cast<unsigned char>(b)));
  }
  Slice s = raw;
  ASSERT_EQ(s.size(), 256u);
  for (int b = 0; b < 256; ++b) {
    EXPECT_EQ(static_cast<unsigned char>(s[static_cast<std::size_t>(b)]),
              static_cast<unsigned char>(b));
  }
}

TEST(Slice, EqualityIsByteWise) {
  std::string a("ab\0c", 4);
  std::string b("ab\0c", 4);
  std::string c("ab\0d", 4);
  EXPECT_EQ(Slice(a), Slice(b));
  EXPECT_NE(Slice(a), Slice(c));
}

TEST(Slice, ConstructsImplicitlyFromStringAndLiteral) {
  std::string owned = "hello";
  Slice from_string = owned;
  Slice from_literal = "hello";
  EXPECT_EQ(from_string, from_literal);
}

}  // namespace
}  // namespace pebbledb
