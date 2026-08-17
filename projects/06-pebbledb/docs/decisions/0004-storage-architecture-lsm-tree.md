# ADR 0004: Storage architecture — LSM tree (spec decision D1)

- **Status:** Accepted
- **Phase:** 0 (foundational — shapes every later phase's structure)
- **Spec decision:** D1 — "LSM vs B+ tree" — default per spec §1.10: **LSM**.

## Context

The spec requires committing to exactly one complete storage engine
architecture before implementation ("a B+ tree is valid only if deliberately
chosen before implementation; never mix two complete engines in MVP" —
§1.3), because the two architectures shape almost everything downstream:
write path, read path, compaction, and file layout are all different
depending on which is chosen. This decision has to be explicit and made
early (Phase 0) even though most of the machinery it implies (MemTable,
SSTable, compaction) doesn't exist until later phases, so that
`docs/architecture.md` and the repository layout (`src/{memtable,sstable,
bloom,cache,manifest,compaction,...}/`) are built toward one target from the
start rather than accreting inconsistently.

## Decision

**Log-structured merge tree (LSM)**, per the spec's recommended default:

- **Write path:** client → WAL append (durability) → active MemTable
  (in-memory, ordered) → immutable MemTable (frozen for flushing) → SSTable
  flush (sorted, immutable on-disk file) → manifest/version metadata update
  → background compaction (merges/reduces SSTables over time).
- **Read path:** active MemTable → immutable MemTable(s) → SSTables from
  newest to oldest (most-recently-flushed first, since newer data shadows
  older data for the same key) → older levels/runs. Bloom filters and
  per-table/block indexes (Phase 5) reduce how many SSTables a read actually
  has to open.
- **Why LSM over a B+ tree** for this project specifically: LSM's write path
  is append-only at every layer (WAL, MemTable inserts, SSTable writes),
  which keeps the crash-recovery story (spec invariant 1, Phase 1's exit
  criterion) simple and matches the portfolio goal of demonstrating
  "write-ahead logging... immutable sorted files... crash recovery and
  background compaction" (spec §1.7 resume narrative) — a B+ tree's
  in-place page updates would need a different durability mechanism
  (shadow paging or a different WAL discipline) to get the same guarantees,
  and would not naturally produce the "SSTable" artifact the roadmap's
  Phase 3 and the inspector tooling (`tools/inspect_sstable`) are built
  around.
- **No B+ tree code exists or will exist in this repository.** Per spec
  §1.3, exactly one complete engine ships in the MVP; there is no partial or
  experimental B+ tree path to keep architecturally separate from the LSM
  code.

## Consequences

- Every later phase's exit criterion in the roadmap (Part 3) is phrased in
  LSM terms (MemTable, SSTable, flush, compaction, manifest) and this ADR is
  the reason why, rather than each phase re-justifying the choice.
- Phase 0's actual code (`DB` backed directly by `std::map`) is explicitly
  *not* the LSM's MemTable yet — it is a reference implementation of the
  public API's put/get/delete/flush/compact/stats semantics with no
  persistence, used to pin correctness (`tests/unit/test_db_map_reference.cpp`)
  before the WAL and MemTable are wired in behind the same API surface. See
  `docs/architecture.md`, "What's implemented so far", for exactly which
  pieces of the diagram above exist today vs. are still roadmap items.
- Repository layout (`src/{wal,memtable,sstable,bloom,cache,manifest,
  compaction,recovery,concurrency,util}/`, spec Part 2) is scaffolded to
  match this architecture; directories for phases not yet started are
  created when that phase begins rather than stubbed out empty in advance.
