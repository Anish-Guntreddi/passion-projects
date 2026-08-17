# PebbleDB architecture

PebbleDB is a log-structured merge tree (LSM) key-value storage engine (ADR
0004, spec decision D1). This document describes the target write/read path
architecture and, separately, exactly what of that architecture is
implemented as of the current phase.

## Target architecture (full roadmap)

```
                              WRITE PATH
                              ----------
  client Put/Delete
        |
        v
  +-----------+     append, checksummed record        +----------------+
  |   WAL     | --------------------------------------> | active        |
  | (Phase 1) |                                          | MemTable     |
  +-----------+                                          | (Phase 2)    |
                                                          +------+-------+
                                                                 | size threshold reached
                                                                 v
                                                          +----------------+
                                                          | immutable      |
                                                          | MemTable       |
                                                          +------+---------+
                                                                 | flush
                                                                 v
                                                          +----------------+     +------------------+
                                                          | SSTable file   | --> | manifest / version|
                                                          | (Phase 3)      |     | metadata (Phase 4)|
                                                          +------+---------+     +------------------+
                                                                 |
                                                                 v background compaction (Phase 6)
                                                          +----------------+
                                                          | merged SSTables|
                                                          +----------------+

                              READ PATH
                              ---------
  client Get(key)
        |
        v
  active MemTable  --miss-->  immutable MemTable(s)  --miss-->  SSTables, newest-first
                                                                       |
                                                        Bloom filter (Phase 5) skips
                                                        tables that provably don't
                                                        have the key; block cache
                                                        (Phase 5) skips re-reading
                                                        blocks already resident.
```

- **Write path:** client → WAL append → active MemTable → immutable MemTable
  → SSTable flush → manifest/version metadata → background compaction (spec
  §1.3).
- **Read path:** active/immutable MemTables → recent SSTables → older
  levels/runs; Bloom filters + indexes reduce file reads (spec §1.3).
- Full rationale for choosing LSM over a B+ tree: ADR 0004.

## What's implemented so far

This section is the authoritative "current state" pointer other files'
comments refer to (e.g. `include/pebbledb/db.hpp`). Update it in the same
change that completes a roadmap phase.

### Phase 0 — Library foundation (complete)

- `Status`/`StatusCode` error model and `Slice` (`std::string_view`) byte-
  ownership rules — ADR 0002.
- `DB` (`include/pebbledb/db.hpp`, `src/db.cpp`): the public
  put/get/delete/flush/compact/stats API (spec §1.2), backed **directly by
  an in-memory `std::map`** — there is no WAL, MemTable, or SSTable wired in
  yet. `Flush()` and `Compact()` are no-ops that return `OK`: there is
  nothing on disk yet to flush or compact.
  - Concurrency: single-writer, multi-reader via one `std::shared_mutex` —
    ADR 0003.
- Build/test/sanitizer scaffolding (CMake + Ninja, GoogleTest, ASan/UBSan/
  TSan build variants) — ADR 0001.
- Byte-encoding toolkit (`util::coding`, fixed-width little-endian
  integers) and `util::Crc32c` (CRC-32C checksum), written as general-
  purpose primitives even though their first consumer is the Phase 1 WAL —
  ADR 0007.
- Exit criterion met: put/get/delete semantics tested without persistence
  (`tests/unit/test_db_map_reference.cpp`, including a 5000-iteration
  randomized-operations-vs-`std::map`-oracle property test).

### Phase 1 — WAL (complete)

- WAL v1 record format: fixed 26-byte header (checksum, magic,
  format_version, op, sequence, key_length, value_length) + variable-length
  key/value payload — full field layout in `docs/file-format.md`; encoding
  rationale in ADR 0007; sequence-number representation in ADR 0005.
- `wal::Writer` (`include/pebbledb/wal/writer.hpp`, `src/wal/writer.cpp`):
  append-only writer with three sync policies (`kEveryWrite`, `kBatched`,
  `kExplicit`) — ADR 0006. Validates key/value size against
  `kMaxKeyLength`/`kMaxValueLength` before ever writing or acknowledging a
  record.
- `wal::Reader` (`include/pebbledb/wal/reader.hpp`, `src/wal/reader.cpp`):
  sequential replay with well-defined truncation (`kTruncated`) vs.
  corruption (`kCorruption`) semantics and no resynchronization past a bad
  record — ADR 0006.
- `util::PosixFile` (`include/pebbledb/util/posix_file.hpp`,
  `src/util/posix_file.cpp`): the WAL's file I/O primitive — sequential
  `write()`/`read()`/`fsync()` on a single fd with an implicit,
  kernel-tracked offset (append-only, replayed front-to-back — no
  `pread`/`pwrite` needed yet; see ADR 0001's scoping note, load-bearing
  once SSTables need random block access in Phase 3). `OpenAppend` also
  `fsync`s the containing directory once, after creating a new WAL file, so
  the directory entry for a brand-new file is itself durable before the
  first record in it can be acknowledged — see `docs/durability.md`.
- Exit criterion met: write sequence replays deterministically after
  simulated crash (`tests/recovery/test_wal_truncation_recovery.cpp`'s
  `Invariant1_...` test, plus an exhaustive per-byte-truncation-offset
  sweep; `tests/corruption/test_wal_corruption.cpp` for invariant 7).

### Not yet implemented (Phases 2–10)

MemTable (ordered structure, sequence/tombstone semantics wired into `DB`),
SSTable (block encoding, writer/reader, index/footer, inspector tool),
flush + manifest, Bloom filter + block cache, compaction, background
workers, crash/corruption harness beyond the WAL layer, benchmark study, and
portfolio hardening (README, diagrams, inspector screenshots) are all
roadmap items per spec Part 3 and not yet started. `DB` does not yet survive
a process restart with data intact — that is Phase 4's exit criterion.

## Module map

| Path | Responsibility |
|---|---|
| `include/pebbledb/status.hpp`, `src/status.cpp` | Error model (ADR 0002) |
| `include/pebbledb/slice.hpp` | Byte-ownership view type (ADR 0002) |
| `include/pebbledb/db.hpp`, `src/db.cpp` | Public API, Phase 0 in-memory reference implementation |
| `include/pebbledb/util/coding.hpp`, `src/util/coding.cpp` | Fixed-width little-endian integer encode/decode (ADR 0007) |
| `include/pebbledb/util/crc32c.hpp`, `src/util/crc32c.cpp` | CRC-32C checksum (ADR 0007) |
| `include/pebbledb/util/posix_file.hpp`, `src/util/posix_file.cpp` | POSIX file I/O primitive for the WAL |
| `include/pebbledb/wal/record.hpp`, `src/wal/record.cpp` | WAL v1 record encode/decode (`docs/file-format.md`, ADR 0005, ADR 0007) |
| `include/pebbledb/wal/writer.hpp`, `src/wal/writer.cpp` | WAL append + sync policy (ADR 0006) |
| `include/pebbledb/wal/reader.hpp`, `src/wal/reader.cpp` | WAL sequential replay (ADR 0006) |
| `tests/unit` | Encoding/checksum/status/slice/DB unit tests |
| `tests/recovery` | Truncation-at-every-offset recovery tests (spec §1.8) |
| `tests/corruption` | Bit-flip / oversized-length corruption tests (invariant 7) |
| `tests/support` | Shared test-only file helpers (`TempFile`, truncate/corrupt byte) |

## See also

- `docs/durability.md` — what "acknowledged" means per sync policy, and the
  boundary between what this project's tests verify and what they cannot
  (real power loss).
- `docs/file-format.md` — the WAL v1 on-disk byte layout, field by field.
- `docs/decisions/` — ADRs for every D1–D9 spec decision affecting a phase
  that has started, plus foundational (non-D-numbered) decisions.
