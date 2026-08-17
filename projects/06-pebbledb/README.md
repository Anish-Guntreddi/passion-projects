# PebbleDB

A miniature C++20 log-structured merge tree (LSM) key-value storage engine,
built up in phases: this repo currently implements **Phase 0 (library
foundation)** and **Phase 1 (WAL)** of the roadmap in the project spec
(`06-pebbledb-spec.md` at the portfolio root). Later phases (MemTable,
SSTable, manifest/flush, Bloom filter + cache, compaction, background
workers, crash/corruption harness, benchmarks, portfolio hardening) are not
yet implemented — see [`docs/architecture.md`](docs/architecture.md)'s
"What's implemented so far" section for the exact current boundary.

## What's here

- **`pebbledb::DB`** (`include/pebbledb/db.hpp`): the public
  put/get/delete/flush/compact/stats API the whole project is built toward.
  Backed today by an in-memory `std::map` — no persistence yet.
  `Flush()`/`Compact()` are no-ops (nothing on disk yet to act on).
- **`pebbledb::wal::Writer` / `Reader`** (`include/pebbledb/wal/`): an
  append-only, checksummed write-ahead log with three sync policies
  (`kEveryWrite`, `kBatched`, `kExplicit`) and deterministic truncation/
  corruption recovery semantics. See
  [`docs/file-format.md`](docs/file-format.md) for the on-disk record
  layout and [`docs/durability.md`](docs/durability.md) for exactly what
  "acknowledged" means per sync policy.
- **`pebbledb::Status`** and **`pebbledb::Slice`**
  (`include/pebbledb/status.hpp`, `include/pebbledb/slice.hpp`): the
  project's error model and byte-ownership rules — see ADR 0002
  (`docs/decisions/0002-status-and-byte-ownership.md`).
- Design decisions are recorded as ADRs in
  [`docs/decisions/`](docs/decisions/) as each is made, before the phase it
  affects begins (spec Part 4 rule 6).

## Build

Requires a C++20 compiler, CMake ≥ 3.20, and Ninja on Linux (the WAL uses
POSIX `open`/`read`/`write`/`fsync` directly). Verified on WSL2 Ubuntu
24.04. Network access is needed once, at configure time, to fetch
GoogleTest via `FetchContent`.

```bash
scripts/build.sh              # Debug build, no sanitizer
scripts/build.sh Debug asan   # Debug + AddressSanitizer + UndefinedBehaviorSanitizer
scripts/build.sh Debug tsan   # Debug + ThreadSanitizer
scripts/build.sh Release none # Release build
```

Binaries land in `build/<BuildType>-<sanitizer>/`.

## Test

```bash
scripts/test.sh               # builds (if needed) then runs the full suite via ctest
scripts/test.sh Debug asan    # same, under ASan+UBSan
scripts/test.sh Debug tsan    # same, under ThreadSanitizer
```

Or directly: `ctest --test-dir build/Debug-none --output-on-failure`.

Three test binaries:
- `pebbledb_unit_tests` — `Status`/`Slice`/byte-encoding/CRC-32C unit
  tests, the Phase 0 in-memory `DB` reference implementation (including a
  concurrent-`Get()` test and a 5000-iteration randomized-operations-vs-
  `std::map`-oracle property test), and pure in-memory WAL record
  encode/decode tests.
- `pebbledb_recovery_tests` — writes a WAL, truncates it at **every**
  possible byte offset, and confirms replay recovers exactly the expected
  prefix deterministically (spec §1.8).
- `pebbledb_corruption_tests` — bit-flip and oversized-declared-length
  corruption of real on-disk WAL files, confirming checksums detect
  corruption rather than silently returning garbage (invariant 7).

## Lint / format

`scripts/lint.sh` (clang-tidy) and `scripts/format.sh` (clang-format) are
not runnable in every local dev environment (see ADR 0001) — CI installs
both tools fresh and runs these exact scripts
(`.github/workflows/ci-pebbledb.yml`).

## What's not here yet

No `tools/pebble_cli` executable yet — Phases 0/1 deliver the library (an
in-memory `DB` reference implementation plus a standalone WAL module) per
the roadmap; a CLI arrives once persistence is wired end-to-end through
`DB`. See `docs/architecture.md` for the full phase-by-phase status.
