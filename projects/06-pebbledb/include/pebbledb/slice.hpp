#pragma once

#include <string>
#include <string_view>

namespace pebbledb {

// Slice: a non-owning, binary-safe view over a byte range. Keys and values
// are "binary-safe bytes" per spec §1.2 — they may contain embedded NUL
// bytes and arbitrary byte values, so a length-prefixed view is used
// instead of a NUL-terminated `const char*`. std::string_view already
// provides exactly this (pointer + length, no ownership, full comparison
// operators, implicit construction from std::string and string literals)
// so PebbleDB aliases it rather than reinventing an equivalent type.
//
// Byte/string ownership rules (Phase 0 deliverable, spec Part 1 §1.2; see
// docs/decisions/0002-status-and-byte-ownership.md for the rationale):
//
//   1. A Slice never owns the memory it points to. It is only valid for as
//      long as the buffer it was constructed from (a std::string, a
//      string literal, a caller-owned char array, ...) is alive and
//      unmodified.
//   2. Every PebbleDB API that *accepts* a Slice (DB::Put, DB::Delete,
//      wal::Record::key/value when being encoded, ...) copies the
//      referenced bytes into its own owned storage (std::string) before
//      returning. The caller is free to destroy or mutate the original
//      buffer immediately after the call returns; PebbleDB never aliases
//      caller-owned memory past the lifetime of a single call.
//   3. Every PebbleDB API that *returns* bytes to the caller (DB::Get's
//      output parameter, a decoded wal::Record's key/value, ...) returns
//      an owned std::string, never a Slice into internal storage. This
//      keeps the ownership boundary at every API edge unambiguous: Slice
//      in, std::string out.
using Slice = std::string_view;

}  // namespace pebbledb
