# PebbleDB — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** PebbleDB — Miniature C++ Storage Engine
**Portfolio position:** 06 of 09 · Track C (C++/Linux systems) · after ArcServe (soft ordering)
**Source of truth:** "06 - PebbleDB - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
PebbleDB is a portfolio-grade key-value storage engine demonstrating C++, persistence, crash recovery, indexing, caching, background work, and storage-performance reasoning. The repo must expose how a write becomes durable bytes and how a lookup traverses memory and disk structures.

### 1.2 Public API (MVP, single-process; durability semantics explicit)
Library + CLI exposing: `put(key, value)`, `get(key)`, `delete(key)`, `flush()`, `compact()` (or background compaction controls), `stats()`. Keys/values are binary-safe bytes.

### 1.3 Storage model
Preferred architecture: **simple LSM tree** (a B+ tree is valid only if deliberately chosen before implementation; never mix two complete engines in MVP).
Write path: client → WAL append → active MemTable → immutable MemTable → SSTable flush → manifest/version metadata → background compaction.
Read path: active/immutable MemTables → recent SSTables → older levels/runs; Bloom filters + indexes reduce file reads.

### 1.4 Functional requirements (MVP)
- **FR1** Append-only WAL with checksummed records; deterministic recovery after unclean shutdown.
- **FR2** In-memory ordered MemTable with sequence/tombstone semantics and size threshold; immutable flush path.
- **FR3** Sorted table (SSTable) file format with documented binary layout (conceptually: data blocks | index block | bloom/filter block | footer/metadata); sparse/block index; per-table (or block-group) Bloom filter; file inspector tools.
- **FR4** Tombstones for delete.
- **FR5** Manifest/version metadata sufficient to recover the active file set; crash-safe updates per the stated durability model.
- **FR6** One understandable compaction strategy (size-tiered or simplified leveled) preserving sequence/tombstone semantics.
- **FR7** Block/page cache (LRU/clock-style).
- **FR8** Background flush/compaction workers **after** the synchronous baseline works, with bounded queues/state coordination and safe shutdown.
- **FR9** Stats/benchmark interface; crash/corruption test harness (kill/restart workloads, truncated WAL/table cases, checksum failures, manifest recovery policy).
- **FR10** All binary formats versioned from the beginning (even if only v1 exists). WAL record header fields: magic/version, operation, sequence number, key length, value length, checksum (exact encoding is a design decision → ADR).

### 1.5 Core invariants (each becomes an explicit test)
1. An acknowledged durable write is recoverable after process crash per the documented sync policy.
2. SSTable contents are immutable after successful creation.
3. Keys inside a sorted table obey comparator order.
4. Tombstone semantics prevent stale older values from resurfacing.
5. Manifest never intentionally references a partially published table.
6. Compaction preserves the latest visible value per key/sequence ordering.
7. Checksums detect corrupted/truncated records rather than silently returning garbage.

### 1.6 Non-goals
SQL; distributed consensus/replication; cross-key transactions; production database compatibility; complex universal/leveled compaction matrices in MVP; lock-free everything; encryption/authentication.

### 1.7 Deliverable artifacts (website/resume)
Write/read path architecture diagram; WAL→MemTable→SSTable lifecycle diagram/animation; crash-recovery sequence; compaction visualization; throughput/latency graphs; read/write/space amplification metrics; inspector screenshots. Resume narrative filled from evidence: *"Built a C++ key-value storage engine with write-ahead logging, in-memory tables, immutable sorted files, Bloom filters, crash recovery and background compaction; benchmarked throughput, tail latency, cache hit rate and write/space amplification."*

### 1.8 Test strategy
Unit: encoding/checksums, comparator, MemTable, Bloom filter, cache, index lookup.
Recovery: write records → truncate at **every possible record boundary** → reopen; acknowledged records survive per sync semantics.
Integration: put/get/delete across flushes; reopen; compaction; cache on/off.
Property: random operation stream vs a `std::map` reference model, across restarts.
Concurrency: foreground read/write during background flush/compaction (TSan).

### 1.9 Benchmark methodology
Separate: operation latency; fsync durability cost; flush/compaction stalls; cold vs warm cache. Record storage device, filesystem, OS, compiler, sync policy, key/value distribution, dataset size. **Never compare results across different durability settings without labeling.** Workloads (Phase 9): write-heavy, read-heavy, mixed, sequential/random keys, cache-hot/cold, overwrite-heavy. Measure throughput, P50/P95/P99, WAL/flush stalls, cache hit rate, bytes written vs logical bytes, disk footprint, write and space amplification.

### 1.10 Open decisions (recommended defaults)
- **D1** LSM vs B+ tree → *default per brief: LSM.*
- **D2** Sequence-number representation → *default: monotonically increasing u64; ADR.*
- **D3** WAL sync policy → *default: configurable {every-write, batched, explicit}; benchmarks label the mode.*
- **D4** MemTable implementation → *default: std::map baseline; skip list is stretch.*
- **D5** Table block size / compression → *default: 4 KB blocks; compression is stretch.*
- **D6** Compaction policy → *default: size-tiered.*
- **D7** Manifest atomicity approach → *default: write-new + fsync + atomic rename; ADR.*
- **D8** Cache key/ownership → *default: (file_id, block_offset) keys, cache owns block copies.*
- **D9** Concurrency model / multiple writers in MVP → *default: single-writer, multi-reader.*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Language | C++20/23 | Brief requirement |
| Build | CMake + Ninja | Brief requirement |
| Testing | GoogleTest/Catch2 | Brief requirement |
| Property/fuzz | Optional libFuzzer/property harness for file/parser code | Brief requirement |
| Sanitizers | ASan, UBSan, TSan | Brief requirement |
| Benchmarking | Google Benchmark or custom workload runner | Brief requirement |
| Analysis | Python for plots | Brief requirement |
| File I/O | Portable pread/pwrite/fsync semantics first; mmap is an experiment, not a prerequisite | Brief requirement |
| CI | Linux build + tests + sanitizer job | Quality bar |

### Repository structure
```
pebbledb/
  CMakeLists.txt
  include/pebbledb/
  src/{api,wal,memtable,sstable,bloom,cache,manifest,compaction,recovery,concurrency,util}/
  tools/{pebble_cli,inspect_sstable,inspect_wal}/
  tests/{unit,recovery,integration,corruption}/
  benchmarks/{workloads,raw,plots}/  benchmarks/methodology.md
  docs/architecture.md  docs/durability.md  docs/file-format.md  docs/compaction.md  docs/decisions/
```

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Library foundation** | Public API, status/error model, byte/string ownership rules, build/tests/sanitizers, in-memory map reference implementation | put/get/delete semantics tested without persistence |
| **1 — WAL** | Record format, append/replay, checksums, truncation behavior, sync policy | Write sequence replays deterministically after simulated crash |
| **2 — MemTable** | Ordered structure, sequence/tombstone semantics, size threshold | Reads merge active state correctly; flush can snapshot immutable state |
| **3 — SSTable** | Block encoding, sorted writer, reader, index/footer, inspector tool | Table written by writer reopens and queries after process restart |
| **4 — Flush + manifest** | Safe immutable-table publication, manifest update, recovered-WAL clearing per design | DB survives restart with data migrated WAL/MemTable → SSTable |
| **5 — Bloom filter + cache** | Negative-read filtering; LRU/clock block cache | Benchmark demonstrates reduced unnecessary reads / cache effects |
| **6 — Compaction** | One simple strategy preserving sequence/tombstone semantics | Duplicate/obsolete records decrease without changing visible state |
| **7 — Background workers** | Flush/compaction off foreground path; bounded coordination; safe shutdown | TSan/stress pass; shutdown drains and stops workers |
| **8 — Crash/corruption testing** | Kill/restart harness, truncated WAL/table cases, checksum failures, manifest recovery | Documented failure behavior is reproducible |
| **9 — Benchmark study** | Workloads + metrics per §1.9 | Committed raw results + plots |
| **10 — Portfolio hardening** | README, format docs, diagrams, benchmark report, inspector screenshots/website assets | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
Skip-list MemTable; prefix/key compression; Snappy/Zstd block compression; mmap read experiment; direct I/O; column families/namespaces; range scans; snapshot/sequence reads; io_uring async reads; custom arena allocator.

### Definition of Done
Persistent put/get/delete survives restart; crash-recovery tests exist; SSTable format documented and inspectable; Bloom/cache and compaction function correctly; background work race-checked; benchmark report quantifies read/write/tail latency and amplification; **no durability claim exceeds tested sync semantics**; README/website explain the complete read/write path.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Every persistence task needs failure/restart acceptance tests.
2. Every binary-format task requires format documentation (`docs/file-format.md` updated in the same change).
3. No complex concurrency or custom allocators before the synchronous engine is correct and recoverable (Phases 0–6 gate Phase 7).
4. Invariants §1.5 are committed as named tests; a change that breaks one is rejected.
5. Never fabricate benchmark numbers; never compare across durability settings unlabeled.
6. Small reviewable commits; ADRs for D1–D9 before the affected phase.

### Kickoff prompt
> Read `06-pebbledb-spec.md` in full. Produce an engineering plan with epics in strict dependency order: API/reference model → WAL → MemTable → SSTable → manifest/flush → filters/cache → compaction → background work → failure testing → benchmarks → portfolio docs. Attach to every persistence task a failure/restart acceptance test, and to every binary-format task a documentation update requirement. Define the status/error model, byte ownership rules, and all v1 binary formats (WAL record, SSTable layout, manifest) as ADRs before implementing them (decisions D1–D9; defaults in §1.10 unless I override). Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (Phases 5, 9)  →  /ship
```
Skip `/qa` (no browser surface). Consider `/cso` once at Phase 8 focused on corruption handling (checksum bypass, truncation edge cases).
