#include "pebbledb/wal/record.hpp"

#include <cstdint>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "pebbledb/util/coding.hpp"
#include "pebbledb/util/crc32c.hpp"

// Pure in-memory encode/decode tests: no filesystem involved. These
// exercise wal::EncodeRecord/DecodeHeader/DecodeBody directly against
// hand-built and round-tripped buffers, which is the right layer to test
// spec §1.8's "encoding/checksums" unit-test line and invariant 7
// ("checksums detect corrupted/truncated records rather than silently
// returning garbage") with fully deterministic, hand-crafted inputs.
// File-based recovery/corruption behavior (real torn writes, real bit
// flips on disk) is covered separately in tests/recovery and
// tests/corruption.

namespace pebbledb::wal {
namespace {

Record MakePut(std::uint64_t seq, std::string key, std::string value) {
  Record r;
  r.sequence = seq;
  r.type = RecordType::kPut;
  r.key = std::move(key);
  r.value = std::move(value);
  return r;
}

Record MakeDelete(std::uint64_t seq, std::string key) {
  Record r;
  r.sequence = seq;
  r.type = RecordType::kDelete;
  r.key = std::move(key);
  return r;
}

// Encodes `record`, then decodes it back and asserts the round trip is
// exact. Returns the encoded bytes so callers can further mutate them for
// corruption tests.
std::string EncodeAndVerifyRoundTrip(const Record& record) {
  std::string encoded;
  EncodeRecord(record, &encoded);
  EXPECT_GE(encoded.size(), kHeaderSize);

  // NOTE: EXPECT_* only (not ASSERT_*) throughout this helper — ASSERT_*
  // expands to a bare `return;` on failure, which does not compile in a
  // function returning non-void. A malformed `header` here would make the
  // DecodeBody call below operate on a default-constructed DecodedHeader,
  // which is fine: it will simply fail its own EXPECT_EQ.
  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);

  std::string_view payload = std::string_view(encoded).substr(kHeaderSize);
  Record decoded;
  EXPECT_EQ(DecodeBody(header, std::string_view(encoded).substr(0, kHeaderSize), payload,
                        &decoded),
            DecodeStatus::kOk);

  EXPECT_EQ(decoded.sequence, record.sequence);
  EXPECT_EQ(decoded.type, record.type);
  EXPECT_EQ(decoded.key, record.key);
  if (record.type == RecordType::kPut) {
    EXPECT_EQ(decoded.value, record.value);
  } else {
    EXPECT_EQ(decoded.value, "");  // Delete records never persist a value
  }
  return encoded;
}

TEST(WalRecordCodec, PutRoundTrip) {
  EncodeAndVerifyRoundTrip(MakePut(1, "hello", "world"));
}

TEST(WalRecordCodec, DeleteRoundTripIgnoresValueField) {
  Record r = MakeDelete(2, "gone");
  r.value = "this-should-never-be-persisted";
  std::string encoded = EncodeAndVerifyRoundTrip(r);
  // value_length field must be 0 for a Delete record on disk.
  DecodedHeader header;
  ASSERT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);
  EXPECT_EQ(header.value_length, 0u);
}

TEST(WalRecordCodec, EmptyKeyAndEmptyValueRoundTrip) {
  EncodeAndVerifyRoundTrip(MakePut(3, "", ""));
}

TEST(WalRecordCodec, BinarySafeKeyAndValueRoundTrip) {
  std::string key("k\0ey", 4);
  std::string value;
  for (int b = 0; b < 256; ++b) {
    value.push_back(static_cast<char>(static_cast<unsigned char>(b)));
  }
  EncodeAndVerifyRoundTrip(MakePut(4, key, value));
}

TEST(WalRecordCodec, LargeKeyAndValueRoundTrip) {
  std::string key(5000, 'k');
  std::string value(70000, 'v');  // exercises length fields beyond one byte
  EncodeAndVerifyRoundTrip(MakePut(5, key, value));
}

TEST(WalRecordCodec, SequenceNumberRoundTripsAtUint64Max) {
  EncodeAndVerifyRoundTrip(
      MakePut(std::numeric_limits<std::uint64_t>::max(), "k", "v"));
}

TEST(WalRecordCodec, MultipleEncodedRecordsConcatenateAndDecodeInOrder) {
  std::string buf;
  EncodeRecord(MakePut(1, "a", "1"), &buf);
  EncodeRecord(MakeDelete(2, "b"), &buf);
  EncodeRecord(MakePut(3, "c", "3"), &buf);

  std::string_view remaining = buf;
  std::uint64_t expected_seq = 1;
  int count = 0;
  while (!remaining.empty()) {
    DecodedHeader header;
    ASSERT_EQ(DecodeHeader(remaining.substr(0, kHeaderSize), &header), DecodeStatus::kOk);
    std::size_t total_len = kHeaderSize + header.key_length + header.value_length;
    Record record;
    ASSERT_EQ(DecodeBody(header, remaining.substr(0, kHeaderSize),
                          remaining.substr(kHeaderSize, header.key_length + header.value_length),
                          &record),
              DecodeStatus::kOk);
    EXPECT_EQ(record.sequence, expected_seq);
    ++expected_seq;
    ++count;
    remaining = remaining.substr(total_len);
  }
  EXPECT_EQ(count, 3);
}

TEST(WalRecordCodec, DecodeHeaderRejectsTruncatedHeader) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);

  DecodedHeader header;
  for (std::size_t len = 0; len < kHeaderSize; ++len) {
    EXPECT_EQ(DecodeHeader(std::string_view(encoded).substr(0, len), &header),
              DecodeStatus::kTruncatedHeader)
        << "len=" << len;
  }
}

TEST(WalRecordCodec, DecodeBodyRejectsTruncatedPayload) {
  std::string encoded;
  EncodeRecord(MakePut(1, "key", "value12345"), &encoded);

  DecodedHeader header;
  ASSERT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);
  std::size_t full_payload_len = header.key_length + header.value_length;
  ASSERT_GT(full_payload_len, 0u);

  for (std::size_t len = 0; len < full_payload_len; ++len) {
    Record record;
    std::string_view short_payload =
        std::string_view(encoded).substr(kHeaderSize, len);
    EXPECT_EQ(DecodeBody(header, std::string_view(encoded).substr(0, kHeaderSize), short_payload,
                          &record),
              DecodeStatus::kTruncatedPayload)
        << "payload_len=" << len;
  }
}

TEST(WalRecordCodec, DecodeHeaderRejectsBadMagic) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);
  // Magic occupies bytes [4, 8).
  encoded[4] ^= 0xFF;

  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kBadMagic);
}

TEST(WalRecordCodec, DecodeHeaderRejectsUnsupportedNewerVersion) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);
  encoded[8] = static_cast<char>(kFormatVersion + 1);  // format_version byte, offset 8

  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kUnsupportedVersion);
}

TEST(WalRecordCodec, DecodeHeaderRejectsZeroVersion) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);
  encoded[8] = 0;  // format_version byte, offset 8

  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kUnsupportedVersion);
}

TEST(WalRecordCodec, DecodeHeaderRejectsOversizedKeyLength) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);
  // key_length occupies bytes [18, 22).
  std::string overwritten = encoded.substr(0, 18);
  util::PutFixed32(&overwritten, kMaxKeyLength + 1);
  overwritten += encoded.substr(22);

  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(overwritten).substr(0, kHeaderSize), &header),
            DecodeStatus::kBadLength);
}

TEST(WalRecordCodec, DecodeHeaderRejectsOversizedValueLength) {
  std::string encoded;
  EncodeRecord(MakePut(1, "k", "v"), &encoded);
  // value_length occupies bytes [22, 26).
  std::string overwritten = encoded.substr(0, 22);
  util::PutFixed32(&overwritten, kMaxValueLength + 1);

  DecodedHeader header;
  EXPECT_EQ(DecodeHeader(std::string_view(overwritten).substr(0, kHeaderSize), &header),
            DecodeStatus::kBadLength);
}

TEST(WalRecordCodec, DecodeBodyRejectsInvalidOperationByte) {
  // Hand-craft a full record with an invalid op byte (0, reserved) but a
  // *correct* checksum for those exact (invalid) bytes — this isolates
  // the operation-byte validation from checksum validation: a record
  // that is internally self-consistent (checksum matches its own
  // content) must still be rejected if the operation byte is not a
  // recognized value.
  std::string body;
  util::PutFixed32(&body, kMagic);
  body.push_back(static_cast<char>(kFormatVersion));
  body.push_back(static_cast<char>(0));  // invalid op byte
  util::PutFixed64(&body, 1);
  util::PutFixed32(&body, 1);  // key_length
  util::PutFixed32(&body, 1);  // value_length
  body += "k";
  body += "v";

  std::string encoded;
  util::PutFixed32(&encoded, util::Crc32c(body));
  encoded += body;

  DecodedHeader header;
  ASSERT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);
  EXPECT_EQ(header.op_byte, 0u);

  Record record;
  EXPECT_EQ(DecodeBody(header, std::string_view(encoded).substr(0, kHeaderSize),
                        std::string_view(encoded).substr(kHeaderSize), &record),
            DecodeStatus::kBadOperation);
}

TEST(WalRecordCodec, DecodeBodyRejectsChecksumMismatchOnCorruptedPayload) {
  std::string encoded;
  EncodeRecord(MakePut(1, "key", "value"), &encoded);
  encoded[kHeaderSize] ^= 0xFF;  // flip a bit in the key payload

  DecodedHeader header;
  ASSERT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);

  Record record;
  EXPECT_EQ(DecodeBody(header, std::string_view(encoded).substr(0, kHeaderSize),
                        std::string_view(encoded).substr(kHeaderSize), &record),
            DecodeStatus::kChecksumMismatch);
}

TEST(WalRecordCodec, DecodeBodyRejectsDirectlyCorruptedChecksumField) {
  std::string encoded;
  EncodeRecord(MakePut(1, "key", "value"), &encoded);
  encoded[0] ^= 0xFF;  // checksum field itself, offset 0

  DecodedHeader header;
  ASSERT_EQ(DecodeHeader(std::string_view(encoded).substr(0, kHeaderSize), &header),
            DecodeStatus::kOk);

  Record record;
  EXPECT_EQ(DecodeBody(header, std::string_view(encoded).substr(0, kHeaderSize),
                        std::string_view(encoded).substr(kHeaderSize), &record),
            DecodeStatus::kChecksumMismatch);
}

TEST(WalRecordCodec, DecodeStatusNameCoversEveryEnumerator) {
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kOk), "Ok");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kTruncatedHeader), "TruncatedHeader");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadMagic), "BadMagic");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kUnsupportedVersion), "UnsupportedVersion");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadLength), "BadLength");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kTruncatedPayload), "TruncatedPayload");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kBadOperation), "BadOperation");
  EXPECT_STREQ(DecodeStatusName(DecodeStatus::kChecksumMismatch), "ChecksumMismatch");
}

}  // namespace
}  // namespace pebbledb::wal
