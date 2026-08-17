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

### Phase 2 — MemTable (complete)

- `memtable::MemTable` (`include/pebbledb/memtable/memtable.hpp`,
  `src/memtable/memtable.cpp`): the LSM's in-memory ordered write buffer
  (spec FR2), backed by `std::map` (spec decision D4's default — ADR
  0008). Each key holds exactly one entry, carrying that write's sequence
  number (ADR 0005) and an explicit `EntryType` (`kValue`/`kTombstone`,
  `include/pebbledb/entry_type.hpp`) — a tombstone is stored, not
  physically erased, so a merge that reaches this layer can distinguish
  "deleted" from "never heard of this key" (invariant 4). Tracks an
  approximate memory-usage total (`ApproximateMemoryUsage()`) against a
  4 MiB default threshold (`kDefaultSizeThresholdBytes`).
- `memtable::MemTableList` (`include/pebbledb/memtable/memtable_list.hpp`,
  `src/memtable/memtable_list.cpp`): owns the active MemTable plus a
  newest-first deque of frozen immutable MemTables, and implements the
  read-side merge across them (active wins outright over any immutable
  layer beneath it, matching the read-path diagram above). `Freeze()`
  snapshots the current active MemTable into the immutable list and
  installs a fresh, empty active MemTable in its place.
- Exit criterion met: reads merge active state correctly, and flush can
  snapshot immutable state (`tests/unit/test_memtable.cpp`,
  `tests/unit/test_memtable_list.cpp`, including
  `Invariant4_TombstoneInActiveShadowsOlderValueInImmutable` and
  `Invariant4_TombstoneInImmutableIsVisibleWhenActiveHasNothing`).

### Phase 3 — SSTable (complete)

- SSTable v1 file format: `[data blocks] [filter block (reserved,
  empty)] [index block] [footer]` — full field layout in
  `docs/file-format.md`; encoding/layout rationale in ADR 0010; block
  size and the reserved filter-block/compression-byte decisions in ADR
  0009 (spec decision D5).
- `sstable::Writer` (`include/pebbledb/sstable/writer.hpp`,
  `src/sstable/writer.cpp`): builds one immutable, sorted table file.
  Rejects an out-of-order or duplicate key (invariant 3) and rejects any
  `Add()`/repeat `Finish()` call after `Finish()` has already completed
  (invariant 2).
- `sstable::Reader` (`include/pebbledb/sstable/reader.hpp`,
  `src/sstable/reader.cpp`): opens a table written by `Writer`, eagerly
  validating the footer and loading the sparse index (FR3) into memory;
  serves `Get()` via one binary search plus one `pread`'d data block, and
  `ForEachEntry()` for full-table iteration. A corrupted block fails only
  the lookups that land in it (no shared sequential cursor, unlike
  `wal::Reader` — see ADR 0010).
- `tools/inspect_sstable`: the file-inspector deliverable (FR3) — prints
  a table's footer, per-block index, and (with `--dump`) every entry.
- `util::PosixFile` gained `ReadAt()`/`Size()` (random-access `pread`/
  `fstat`) alongside its existing sequential `Read()`/`Append()` — the
  pread capability ADR 0001 anticipated becoming load-bearing at this
  phase. It also gained `OpenNew()`, used by `sstable::Writer::Open()`
  instead of the WAL's reuse/resume-friendly `OpenAppend()` — a table
  file must never be silently appended to or reused, since a `Writer`'s
  offset bookkeeping starts from 0 and must exactly match on-disk
  positions; `OpenNew()` fails if `path` already has real content, while
  still accepting an absent or already-empty path (ADR 0011).
- `sstable::Reader::Open()` and every `ReadBlock()` call validate a
  `BlockHandle`'s `offset + size` against the file's actual size before
  issuing the corresponding `pread`, so a corrupted-but-checksum-valid
  footer or index entry declaring an out-of-range block cannot drive an
  unbounded allocation/read (invariant 7; ADR 0011).
- Exit criterion met: a table written by `Writer` reopens and queries
  correctly after the writing process's `Writer` object is destroyed and
  a brand-new `Reader` opens the same path
  (`tests/integration/test_sstable_writer_reader.cpp`), including across
  many data blocks, tombstones, and an empty (zero-entry) table.
  `tests/integration/test_memtable_flush_to_sstable.cpp` additionally
  demonstrates Phase 2 and Phase 3 composing end-to-end: a frozen
  MemTable's `entries()` flushed straight into a `Writer` reproduces
  exactly the same state once reopened through a `Reader`, tombstones
  included. `tests/corruption/test_sstable_corruption.cpp` covers
  invariant 7 for this format (footer/index/data-block corruption and
  truncation).

### Not yet implemented (Phases 4–10)

Flush + manifest (wiring MemTable/SSTable into `DB`'s write/read path with
crash-safe manifest updates), Bloom filter + block cache, compaction,
background workers, crash/corruption harness beyond the WAL/SSTable
layers, benchmark study, and portfolio hardening (README, diagrams,
inspector screenshots) are all roadmap items per spec Part 3 and not yet
started. `DB` (`include/pebbledb/db.hpp`) is still exactly its Phase 0
in-memory-`std::map` reference implementation — it does not yet use the
WAL, MemTable, or SSTable modules at all, and does not yet survive a
process restart with data intact. That wiring, and the manifest that makes
it crash-safe, is Phase 4's exit criterion.

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
| `include/pebbledb/entry_type.hpp`, `src/entry_type.cpp` | Shared value/tombstone vocabulary for MemTable + SSTable (ADR 0008) |
| `include/pebbledb/lookup_result.hpp`, `src/lookup_result.cpp` | Shared three-state (found/deleted/not-found) lookup result for MemTable/MemTableList/SSTable Reader |
| `include/pebbledb/memtable/memtable.hpp`, `src/memtable/memtable.cpp` | Ordered in-memory write buffer (ADR 0008) |
| `include/pebbledb/memtable/memtable_list.hpp`, `src/memtable/memtable_list.cpp` | Active/immutable MemTable stack + read merge |
| `include/pebbledb/sstable/format.hpp`, `src/sstable/format.cpp` | SSTable v1 block/index/footer encode/decode (`docs/file-format.md`, ADR 0009, ADR 0010) |
| `include/pebbledb/sstable/writer.hpp`, `src/sstable/writer.cpp` | SSTable sorted writer (invariants 2, 3) |
| `include/pebbledb/sstable/reader.hpp`, `src/sstable/reader.cpp` | SSTable point lookup + full-table iteration |
| `tools/inspect_sstable` | SSTable file inspector (FR3) |
| `tests/unit` | Encoding/checksum/status/slice/DB/MemTable/SSTable-format unit tests |
| `tests/recovery` | Truncation-at-every-offset WAL recovery tests (spec §1.8) |
| `tests/integration` | Real-file SSTable writer/reader reopen tests; MemTable-to-SSTable flush end-to-end tests |
| `tests/corruption` | Bit-flip / oversized-length / truncation corruption tests (invariant 7) for the WAL and SSTable formats |
| `tests/support` | Shared test-only file helpers (`TempFile`, truncate/corrupt byte) |

## See also

- `docs/durability.md` — what "acknowledged" means per sync policy, and the
  boundary between what this project's tests verify and what they cannot
  (real power loss).
- `docs/file-format.md` — the WAL v1 and SSTable v1 on-disk byte layouts,
  field by field.
- `docs/decisions/` — ADRs for every D1–D9 spec decision affecting a phase
  that has started, plus foundational (non-D-numbered) decisions.
