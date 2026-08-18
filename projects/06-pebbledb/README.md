# PebbleDB

A miniature C++20 log-structured merge tree (LSM) key-value storage
engine, built up in phases: this repo currently implements **Phase 0
(library foundation)** through **Phase 5 (Bloom filter + block cache)**
of the roadmap in the project spec (`06-pebbledb-spec.md` at the
portfolio root). Later phases (compaction, background workers,
crash/corruption harness beyond what's already covered, benchmarks,
portfolio hardening) are not yet implemented — see
[`docs/architecture.md`](docs/architecture.md)'s "What's implemented so
far" section for the exact current boundary.

As of Phase 4, `pebbledb::DB` is a real, disk-backed storage engine:
`DB::Open(dbname, &db)` opens (or creates) a directory-rooted database
wired straight through WAL → MemTable → SSTable → manifest exactly per
`docs/architecture.md`'s write/read path diagram, and survives a full
process restart with data intact. `DB`'s original in-memory-only mode
(the default constructor, no path) remains fully supported — see
[ADR 0013](docs/decisions/0013-flush-and-recovery-design.md) for why
both modes are real, not one a stepping stone to the other.

## What's here

- **`pebbledb::DB`** (`include/pebbledb/db.hpp`): the public
  put/get/delete/flush/compact/stats API the whole project is built
  toward.
  - **Persistent mode** (`DB::Open(dbname, &db)`): a directory-rooted,
    crash-recoverable engine. `Put`/`Delete` append a checksummed WAL
    record, then apply to the active MemTable; crossing a size threshold
    (or an explicit `Flush()` call) synchronously freezes the active
    MemTable, writes it to a new SSTable, atomically publishes it via
    the manifest, rotates to a new WAL segment, and deletes the
    now-superseded old one. `Get()` merges across the active/immutable
    MemTables, then every SSTable newest-first, consulting each table's
    Bloom filter and the shared block cache along the way. `Open()`
    recovers prior state by loading the manifest and replaying whatever
    WAL it points at. See [ADR 0013](docs/decisions/0013-flush-and-recovery-design.md).
  - **In-memory-only mode** (`DB db;`, no path): backed by
    `memtable::MemTableList` alone — no WAL, no SSTables, no manifest,
    no persistence across a restart. A fully supported second mode, not
    a deprecated one.
  - Concurrency: single-writer, multi-reader (ADR 0003).
- **`pebbledb::wal::Writer` / `Reader`** (`include/pebbledb/wal/`): an
  append-only, checksummed write-ahead log with three sync policies
  (`kEveryWrite`, `kBatched`, `kExplicit`) and deterministic truncation/
  corruption recovery semantics. See
  [`docs/file-format.md`](docs/file-format.md) for the on-disk record
  layout and [`docs/durability.md`](docs/durability.md) for exactly what
  "acknowledged" means per sync policy.
- **`pebbledb::memtable::MemTable` / `MemTableList`**
  (`include/pebbledb/memtable/`): the LSM's in-memory ordered write
  buffer, with sequence-numbered, tombstone-aware entries (spec FR2, FR4)
  and an active/immutable-layer read merge (`MemTableList::Get`). See ADR
  0008.
- **`pebbledb::sstable::Writer` / `Reader`** (`include/pebbledb/sstable/`):
  builds and reads immutable, sorted, checksummed on-disk table files —
  data blocks, a per-table Bloom filter block (Phase 5, on by default),
  a sparse in-memory index, and a fixed footer. See
  [`docs/file-format.md`](docs/file-format.md) for the on-disk layout,
  ADR 0009/ADR 0010 for the Phase 3 design rationale, and ADR 0014 for
  the Phase 5 Bloom filter. `Reader` optionally reads through a shared
  `cache::BlockCache` (see below) and exposes `filter_checks()`/
  `filter_rejections()`/`cache_hits()`/`cache_misses()`/
  `data_blocks_read()` so "reduced unnecessary reads" is directly
  measurable. `tools/inspect_sstable` inspects a table file from the
  command line (`inspect_sstable <path> [--dump]`).
- **`pebbledb::manifest`** (`include/pebbledb/manifest/`): the v1
  binary format recording the current file-number counter, the WAL
  segment (if any) still needing replay, the highest known-durable
  sequence number, and the list of currently-published SSTables
  (`format.hpp`/`format.cpp`), plus the crash-safe load/save I/O layer
  implementing spec decision D7 — write-new + fsync + atomic rename
  (`manifest.hpp`/`manifest.cpp`). See ADR 0012.
- **`pebbledb::bloom`** (`include/pebbledb/bloom/`): standard Bloom
  filter build/query (default 10 bits/key, ~1% false-positive rate),
  used by `sstable::Writer`/`Reader`. See ADR 0014.
- **`pebbledb::cache::BlockCache`** (`include/pebbledb/cache/`): a
  shared LRU cache of decoded SSTable data blocks, keyed by
  `(file_id, block_offset)` (spec decision D8) and shared across every
  `sstable::Reader` a `DB` opens. See ADR 0015.
- **`pebbledb::Status`** and **`pebbledb::Slice`**
  (`include/pebbledb/status.hpp`, `include/pebbledb/slice.hpp`): the
  project's error model and byte-ownership rules — see ADR 0002
  (`docs/decisions/0002-status-and-byte-ownership.md`).
- Design decisions are recorded as ADRs in
  [`docs/decisions/`](docs/decisions/) as each is made, before the phase it
  affects begins (spec Part 4 rule 6).

## Build

Requires a C++20 compiler, CMake ≥ 3.20, and Ninja on Linux (WAL/SSTable/
manifest I/O uses POSIX `open`/`read`/`write`/`pread`/`fsync`/`rename`
directly). Verified on WSL2 Ubuntu 24.04. Network access is needed once,
at configure time, to fetch GoogleTest via `FetchContent`.

```bash
scripts/build.sh              # Debug build, no sanitizer
scripts/build.sh Debug asan   # Debug + AddressSanitizer + UndefinedBehaviorSanitizer
scripts/build.sh Debug tsan   # Debug + ThreadSanitizer
scripts/build.sh Release none # Release build
```

Binaries land in `build/<BuildType>-<sanitizer>/`; `tools/inspect_sstable`
lands in `build/<BuildType>-<sanitizer>/tools/`.

## Test

```bash
scripts/test.sh               # builds (if needed) then runs the full suite via ctest
scripts/test.sh Debug asan    # same, under ASan+UBSan
scripts/test.sh Debug tsan    # same, under ThreadSanitizer
```

Or directly: `ctest --test-dir build/Debug-none --output-on-failure`.

Four test binaries (231 tests total, green under Debug-none, Debug-asan,
Debug-tsan, and Release-none):

- `pebbledb_unit_tests` — `Status`/`Slice`/byte-encoding/CRC-32C unit
  tests, `DB`'s in-memory-mode reference tests (including a concurrent-
  `Get()` test and a 5000-iteration randomized-operations-vs-`std::map`-
  oracle property test), pure in-memory WAL/SSTable/manifest record
  encode/decode tests, `MemTable`/`MemTableList` unit tests (including a
  5000-iteration randomized property test), and pure in-memory Bloom
  filter (`BuildFilter`/`KeyMayMatch`, including a no-false-negatives
  property test) and `BlockCache` (LRU eviction order, capacity
  invariants, concurrent-access) unit tests.
- `pebbledb_recovery_tests` — writes a WAL, truncates it at **every**
  possible byte offset, and confirms replay recovers exactly the
  expected prefix deterministically (spec §1.8); fault-injects
  `DB::Flush()` failures at each invariant-5-relevant boundary (SSTable
  creation, manifest publish) and confirms no data is ever lost.
- `pebbledb_integration_tests` — writes real SSTable files, destroys the
  `Writer`, reopens a brand-new `Reader` on the same path, and confirms
  every lookup resolves correctly (spec Phase 3 exit criterion), now
  including a real, non-empty Bloom filter by default; an end-to-end
  MemTable-freeze-then-flush-then-reopen test; filter/cache
  reduced-unnecessary-reads tests (Phase 5 exit criterion, measured
  directly via `sstable::Reader`'s counters, paired with disabled/no-cache
  controls); and a real, disk-backed `DB` restart/persistence suite
  (Phase 4 exit criterion) including a randomized-operations-vs-oracle
  property test that restarts every few hundred operations (spec §1.8's
  "across restarts" requirement).
- `pebbledb_corruption_tests` — bit-flip, truncation, and oversized-
  declared-length corruption of real on-disk WAL, SSTable, **and
  manifest** files, confirming checksums detect corruption rather than
  silently returning garbage (invariant 7), plus `DB::Open()` failing
  cleanly (not crashing, not silently discarding data) when a manifest
  references a missing SSTable or WAL file.

## Lint / format

`scripts/lint.sh` (clang-tidy) and `scripts/format.sh` (clang-format) are
not runnable in every local dev environment (see ADR 0001) — CI installs
both tools fresh and runs these exact scripts
(`.github/workflows/ci-pebbledb.yml`).

## What's not here yet

Compaction (Phase 6), background flush/compaction workers (Phase 7 — all
flushing today is synchronous and foreground, by design at this phase —
see ADR 0013), a crash/corruption harness beyond what Phases 1–5 already
cover, a benchmark study (Phase 9), and portfolio hardening (Phase 10:
diagrams, benchmark report, inspector screenshots) are all upcoming
per the roadmap. `tools/pebble_cli` (an end-to-end CLI) does not exist
yet either. See `docs/architecture.md` for the full phase-by-phase
status.
