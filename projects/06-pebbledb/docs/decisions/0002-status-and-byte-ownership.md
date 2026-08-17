# ADR 0002: Status/error model and byte ownership rules

- **Status:** Accepted
- **Phase:** 0 (library foundation)
- **Related spec decisions:** Part 1 §1.2 (Phase 0 deliverable: "status/error
  model, byte/string ownership rules")

## Context

Every fallible PebbleDB operation (missing key, truncated WAL record, disk
full, bad argument) needs a uniform way to report *which* failure occurred,
and every API boundary that hands bytes across (keys, values, WAL record
fields) needs an unambiguous rule for who owns that memory and for how long.
Both are Phase 0 deliverables per the spec and both are foundational: every
later phase's code depends on getting these two rules right once rather than
re-deciding them per module.

## Decision — error model

- **`Status`, not exceptions**, for every fallible operation in the public
  API (`include/pebbledb/status.hpp`). Storage engines fail routinely
  (missing key, truncated record, disk full) in ways callers up and down the
  write/read path need to branch on by *kind*, which a single unstructured
  exception type does not make convenient, and which stack-unwinding
  exceptions make harder to reason about across the WAL/recovery boundary
  where partial completion is common and expected, not exceptional.
- **A closed set of status codes** (`StatusCode`): `kOk`, `kNotFound`,
  `kCorruption`, `kIOError`, `kInvalidArgument`, `kNotSupported`. Each has a
  static factory (`Status::NotFound(...)`, `Status::IOError(...)`, ...) that
  pairs the code with an optional human-readable message; `Status::OK()` (or
  the default constructor) is always available with no message.
- **Default-constructed `Status` is OK.** The happy path costs nothing to
  spell: a function that always succeeds can `return Status();` or fall
  through a default member without importing any additional ceremony.
- **`Status` is returned by value**, not via output parameter or global
  `errno`-style state, so it composes normally with early-return control
  flow (`if (Status s = Foo(); !s.ok()) return s;`) and is safe to store,
  copy, and inspect after the call that produced it returns.
- `Status::ToString()` renders `"OK"` for the OK status, otherwise
  `"<Kind>: <message>"` (or just `"<Kind>"` if the message is empty) — used
  by tests and diagnostics, never parsed back programmatically.

## Decision — byte/string ownership rules

- **`Slice = std::string_view`** (`include/pebbledb/slice.hpp`): a
  non-owning, binary-safe view over a byte range. Keys and values are
  "binary-safe bytes" per spec §1.2 (they may contain embedded NUL bytes and
  arbitrary byte values), so a length-prefixed view is required instead of a
  NUL-terminated `const char*`; `std::string_view` already provides exactly
  that (pointer + length, full comparator support, implicit construction
  from `std::string` and string literals), so PebbleDB aliases it rather
  than reinventing an equivalent type.
- **Rule 1 — a `Slice` never owns memory.** It is only valid for as long as
  the buffer it was constructed from (a `std::string`, a string literal, a
  caller-owned array, ...) is alive and unmodified. PebbleDB never
  constructs a `Slice` over memory it manages the lifetime of and then hands
  that `Slice` back across an API boundary.
- **Rule 2 — every API that *accepts* a `Slice` copies it before returning.**
  `DB::Put`, `DB::Delete`, and `wal::EncodeRecord` (via `Record::key`/
  `Record::value`, which are owned `std::string`s) all copy the referenced
  bytes into their own storage before the call returns. The caller is free
  to destroy or mutate the original buffer immediately afterward; PebbleDB
  never aliases caller-owned memory past the lifetime of a single call.
- **Rule 3 — every API that *returns* bytes returns an owned `std::string`,
  never a `Slice` into internal storage.** `DB::Get`'s output parameter and
  a decoded `wal::Record`'s `key`/`value` are always `std::string`. This
  keeps the ownership boundary at every API edge unambiguous in one
  direction only: `Slice` in, `std::string` out. A caller can never be
  handed a view whose backing storage might be mutated or freed by a
  subsequent PebbleDB call (e.g. a second `Get()` overwriting a MemTable
  slot).

## Consequences

- Callers write straight-line code (`if (!status.ok()) { ... }`) instead of
  `try`/`catch`, and every fallible signature in the codebase is
  self-documenting about failure via its `Status` return.
- The copy-in/copy-out ownership rule costs an extra allocation and copy on
  every `Put`/`Get`, which is the correct trade at PebbleDB's scope
  (portfolio-grade correctness and clarity over the last few percent of
  throughput) and is revisited only if benchmarking (Phase 9) shows it is
  the dominant cost — not before.
- `Slice`'s validity window matching "one call" makes it straightforward to
  reason about aliasing bugs: `tests/unit/test_db_map_reference.cpp`'s
  `PutCopiesBytesRatherThanAliasingCallerBuffer` and
  `GetCopiesIntoOutputRatherThanAliasingInternalStorage` tests exist
  specifically to pin this contract with ASan/UBSan builds able to catch a
  regression (a use-after-free of the caller's buffer, or of DB-internal
  storage) if it were ever violated.
