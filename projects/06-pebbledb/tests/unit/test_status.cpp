#include "pebbledb/status.hpp"

#include <gtest/gtest.h>

namespace pebbledb {
namespace {

TEST(Status, DefaultConstructedIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::kOk);
  EXPECT_EQ(s.ToString(), "OK");
}

TEST(Status, OkFactory) {
  Status s = Status::OK();
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.ToString(), "OK");
}

TEST(Status, NotFoundWithoutMessage) {
  Status s = Status::NotFound();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsNotFound());
  EXPECT_FALSE(s.IsCorruption());
  EXPECT_EQ(s.code(), StatusCode::kNotFound);
  EXPECT_EQ(s.ToString(), "NotFound");
}

TEST(Status, NotFoundWithMessage) {
  Status s = Status::NotFound("key 'foo'");
  EXPECT_TRUE(s.IsNotFound());
  EXPECT_EQ(s.message(), "key 'foo'");
  EXPECT_EQ(s.ToString(), "NotFound: key 'foo'");
}

TEST(Status, Corruption) {
  Status s = Status::Corruption("checksum mismatch");
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsCorruption());
  EXPECT_EQ(s.ToString(), "Corruption: checksum mismatch");
}

TEST(Status, IOError) {
  Status s = Status::IOError("disk full");
  EXPECT_TRUE(s.IsIOError());
  EXPECT_EQ(s.ToString(), "IOError: disk full");
}

TEST(Status, InvalidArgument) {
  Status s = Status::InvalidArgument("batch_size must be >= 1");
  EXPECT_TRUE(s.IsInvalidArgument());
  EXPECT_EQ(s.ToString(), "InvalidArgument: batch_size must be >= 1");
}

TEST(Status, NotSupported) {
  Status s = Status::NotSupported("range scans");
  EXPECT_TRUE(s.IsNotSupported());
  EXPECT_EQ(s.ToString(), "NotSupported: range scans");
}

TEST(Status, PredicatesAreMutuallyExclusive) {
  Status s = Status::Corruption("x");
  EXPECT_FALSE(s.ok());
  EXPECT_FALSE(s.IsNotFound());
  EXPECT_TRUE(s.IsCorruption());
  EXPECT_FALSE(s.IsIOError());
  EXPECT_FALSE(s.IsInvalidArgument());
  EXPECT_FALSE(s.IsNotSupported());
}

TEST(Status, CopyableAndAssignable) {
  Status a = Status::IOError("boom");
  Status b = a;
  EXPECT_EQ(a.ToString(), b.ToString());
  Status c;
  c = a;
  EXPECT_TRUE(c.IsIOError());
}

}  // namespace
}  // namespace pebbledb
