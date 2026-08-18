#include "pebbledb/bloom/bloom_filter.hpp"

#include "pebbledb/util/crc32c.hpp"

namespace pebbledb::bloom {

namespace {

// h1: the filter's single real hash evaluation per key. Reuses
// util::Crc32c (already implemented, tested, and used everywhere else in
// this codebase for on-disk checksums) rather than introducing a second,
// bespoke hash function -- CRC-32C's avalanche behavior is good enough
// for bit-selection purposes even though checksumming, not hashing, is
// its primary role here.
std::uint32_t Hash1(std::string_view key) { return util::Crc32c(key); }

// h2: derived cheaply from h1 via a bit rotation, rather than computing
// a second, independent hash from scratch. This is the standard
// Kirsch/Mitzenmacher "double hashing" construction for Bloom filters
// (Kirsch & Mitzenmacher, "Less Hashing, Same Performance: Building a
// Better Bloom Filter", 2006): probe i's bit position is
// g_i(x) = h1(x) + i * h2(x) mod num_bits, which gives k
// practically-independent probe positions from just one real hash
// evaluation per key, with (per that paper) the same asymptotic
// false-positive rate as k fully independent hash functions.
std::uint32_t Hash2(std::uint32_t h1) { return (h1 >> 15) | (h1 << 17); }

// k = bits_per_key * ln(2), the standard optimal-number-of-probes
// formula for a Bloom filter sized at `bits_per_key` bits per element
// (see ADR 0014 for the false-positive-rate math this is derived from).
// Rounded to the nearest integer and clamped to [1, 30] -- the same
// clamp range used by essentially every mainstream Bloom filter
// implementation, to avoid a degenerate k=0 filter (which would report
// every key as "maybe present", i.e. no filtering at all) or an
// impractically large number of probes per lookup for a very large
// `bits_per_key` input.
std::size_t ChooseNumProbes(std::size_t bits_per_key) {
  const double k = static_cast<double>(bits_per_key) * 0.69314718055994530942;  // ln(2)
  auto probes = static_cast<std::size_t>(k + 0.5);
  if (probes < 1) {
    probes = 1;
  }
  if (probes > 30) {
    probes = 30;
  }
  return probes;
}

}  // namespace

std::string BuildFilter(const std::vector<std::string>& keys, const FilterPolicy& policy) {
  if (keys.empty() || policy.bits_per_key == 0) {
    return {};
  }

  const std::size_t num_probes = ChooseNumProbes(policy.bits_per_key);

  std::size_t num_bits = keys.size() * policy.bits_per_key;
  if (num_bits < 64) {
    // A tiny filter (very few keys) would otherwise have an
    // unnecessarily high false-positive rate; 64 bits is a small,
    // fixed floor, not a spec requirement -- see ADR 0014.
    num_bits = 64;
  }
  const std::size_t num_bytes = (num_bits + 7) / 8;
  num_bits = num_bytes * 8;  // round up to a whole byte count

  std::string result;
  result.reserve(1 + num_bytes);
  result.push_back(static_cast<char>(static_cast<std::uint8_t>(num_probes)));
  result.append(num_bytes, '\0');
  char* bits = result.data() + 1;

  for (const std::string& key : keys) {
    const std::uint32_t h1 = Hash1(key);
    const std::uint32_t h2 = Hash2(h1);
    std::uint32_t g = h1;
    for (std::size_t i = 0; i < num_probes; ++i) {
      const std::uint32_t bit_pos = g % static_cast<std::uint32_t>(num_bits);
      bits[bit_pos / 8] =
          static_cast<char>(static_cast<unsigned char>(bits[bit_pos / 8]) | (1u << (bit_pos % 8)));
      g += h2;  // wrapping uint32 addition is fine here -- see Hash2's comment
    }
  }
  return result;
}

bool KeyMayMatch(std::string_view key, std::string_view filter) {
  // Fewer than 2 bytes cannot even hold a valid num_probes byte plus a
  // non-empty bit array -- degrade to "may match" (do the real lookup)
  // rather than ever risking a false negative from a malformed/corrupted
  // filter -- see the header comment.
  if (filter.size() < 2) {
    return true;
  }

  const std::size_t num_probes = static_cast<std::uint8_t>(filter[0]);
  const std::string_view bits = filter.substr(1);
  const std::size_t num_bits = bits.size() * 8;

  const std::uint32_t h1 = Hash1(key);
  const std::uint32_t h2 = Hash2(h1);
  std::uint32_t g = h1;
  for (std::size_t i = 0; i < num_probes; ++i) {
    const std::uint32_t bit_pos = g % static_cast<std::uint32_t>(num_bits);
    const auto byte = static_cast<unsigned char>(bits[bit_pos / 8]);
    if ((byte & (1u << (bit_pos % 8))) == 0) {
      return false;  // definitely absent
    }
    g += h2;
  }
  return true;  // may be present
}

}  // namespace pebbledb::bloom
