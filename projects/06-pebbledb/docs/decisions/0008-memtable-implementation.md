# ADR 0008: MemTable implementation (spec decision D4)

- **Status:** Accepted
- **Phase:** 2 (MemTable)
- **Spec decision:** D4 — "MemTable implementation" — default per spec
  §1.10: **`std::map` baseline; skip list is stretch.**

## Context

The LSM write path (ADR 0004) needs an in-memory, ordered write buffer
(spec FR2) that (a) supports point Put/Delete/Get, (b) iterates in
ascending key order so a Phase 3 SSTable writer can consume it directly
(invariant 3), and (c) tracks its own approximate size so a caller can
decide when to freeze it and start a flush (spec FR2, "size threshold").
Spec Part 3's stretch-goal list explicitly separates a skip-list MemTable
out as post-MVP, so the Phase 2 implementation has to be useful and
correct without one.

## Decision — `std::map`, one entry per key

`memtable::MemTable` is backed by `std::map<std::string, Entry,
std::less<>>` (`include/pebbledb/memtable/memtable.hpp`), the spec's
default. A red-black tree gives ordered iteration and O(log n) point
operations for free from the standard library, with none of a skip
list's custom allocator/node-linking code to write and sanitizer-harden
first — exactly the right trade for a phase whose job is "prove the
MemTable's *semantics*", not "prove a lock-free/probabilistic data
structure's concurrency properties" (which is out of scope until, at
minimum, Phase 7 per spec Part 4 rule 3).

**Each key holds exactly one entry** — the most recent Put/Delete for
that key overwrites whatever was there before, carrying forward that
write's sequence number and an explicit `EntryType` (`kValue` or
`kTombstone`, `include/pebbledb/entry_type.hpp`). This is deliberately
*not* a multi-version structure that retains every historical write for a
key (the way LevelDB's memtable does, to support snapshot reads): spec
§1.6 lists "snapshot/sequence reads" as a stretch goal, not an MVP
requirement, so there is no reader that ever needs to see an older
version of a key once a newer write for that key has landed in the same
MemTable. Collapsing to one entry per key keeps `ApproximateMemoryUsage()`
bounded by the *live* keyspace rather than by total write volume, and
keeps `entries()` — the view a flush iterates — exactly as large as the
final on-disk SSTable's row count needs to be.

**Tombstones are stored, not erased** — see memtable.hpp's class comment
for the full rationale. In short: a MemTable is one layer in a stack
(active → immutable MemTable(s) → SSTables, newest first —
`docs/architecture.md`), and `memtable::MemTableList`'s merge logic
(`memtable_list.hpp`) depends on being able to distinguish "this layer
has a tombstone for this key" (stop searching older layers — invariant 4)
from "this layer has never heard of this key" (keep searching). Physically
erasing a deleted key would collapse those two states into one and let a
stale value from an older layer resurface.

## Decision — size threshold: 4 MiB default, approximate accounting

`MemTable::kDefaultSizeThresholdBytes = 4 MiB`. `ApproximateMemoryUsage()`
tracks, incrementally (O(1) per Put/Delete, not a re-sum of the whole
map), `sum(key.size() + value.size() + kPerEntryOverheadBytes)` across
every entry (`kPerEntryOverheadBytes = 32`, a rough stand-in for
`std::map`'s actual red-black-tree node header + allocator bookkeeping,
which varies by standard-library implementation and is not worth modeling
exactly at this project's scope). This is explicitly an *approximation*,
matching spec FR2's plain "size threshold" wording rather than an exact
byte-for-byte accounting requirement — the threshold's job is to bound how
much unflushed data can be lost on crash and to keep flush latency
predictable, not to be a precise memory profiler. 4 MiB is a conventional
LSM default (matches common real-world engines' defaults at a similar
scope) and is small enough that `tests/unit/test_memtable.cpp` and
`tests/integration/test_memtable_flush_to_sstable.cpp`'s randomized tests
run in well under a second without needing a smaller threshold override.

Sequence-number allocation is out of scope for `MemTable` itself, exactly
as ADR 0005 already scoped it for the WAL: `Put`/`Delete` take a
caller-supplied `sequence`, and do not themselves maintain a counter. A
single global counter is introduced when the WAL, MemTable, and DB are
wired together end-to-end (Phase 4).

## Consequences

- `MemTable::entries()` is always in ascending key order by construction
  (an `std::map`'s in-order iteration), which is exactly the order
  `sstable::Writer::Add()` requires (invariant 3) — no separate sort step
  is ever needed between "freeze a MemTable" and "flush it to an
  SSTable" (demonstrated end-to-end by
  `tests/integration/test_memtable_flush_to_sstable.cpp`).
- `MemTable::Get()` and `MemTableList::Get()` return the three-state
  `LookupResult` (`kFound`/`kDeleted`/`kNotFound`,
  `include/pebbledb/lookup_result.hpp`) rather than collapsing "deleted"
  into "not found" the way `DB::Get`'s Phase-0 `Status` does — every
  intermediate LSM layer needs that distinction; only the *outermost*
  caller (Phase 4's `DB::Get`, once wired) collapses it exactly once, at
  the top of the merged read path.
- A future skip-list MemTable (stretch goal, spec Part 3) would replace
  `MemTable`'s internal storage but not its public API (`Put`/`Delete`/
  `Get`/`entries()`/`ApproximateMemoryUsage()`), since nothing outside
  `MemTable` depends on `std::map` specifically — `MemTableList` and any
  future flush code only ever go through that API surface.
- `EntryType` (spec FR4, invariant 4) is defined once, at
  `pebbledb::` root (`include/pebbledb/entry_type.hpp`), and reused
  unchanged by both `MemTable` (this ADR) and `sstable` (ADR 0009/0010) —
  a flush passes an entry from one layer to the other with no semantic
  transformation, so the two layers deliberately share one vocabulary
  rather than each defining an equivalent-but-distinct enum. `wal::
  RecordType` (ADR 0007) stays separate on purpose: it represents "an
  operation that was requested", not "the LSM's current resolved state
  for a key", and code that replays a WAL record into a MemTable (Phase 4)
  maps between the two explicitly at that one call site.
