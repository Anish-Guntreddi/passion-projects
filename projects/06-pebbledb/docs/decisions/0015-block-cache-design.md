# ADR 0015: Block cache design (spec decision D8)

- **Status:** Accepted
- **Phase:** 5 (Bloom filter + block cache)
- **Spec requirement:** FR7 — "Block/page cache (LRU/clock-style)."
  Roadmap Phase 5 deliverable: "LRU/clock block cache." Exit criterion:
  "Benchmark demonstrates reduced unnecessary reads / cache effects."
- **Spec decision:** D8 — "Cache key/ownership" — default per spec
  §1.10: **`(file_id, block_offset)` keys, cache owns block copies.**

## Context

Every `sstable::Reader::Get()`/`ForEachEntry()` call that misses the
Bloom filter (or that filter is disabled/absent for) reads its data
block from disk via `pread` (`ReadBlock()`). A repeated lookup of the
same or a nearby key — a realistic access pattern (hot keys, range-like
access, a benchmark replaying a skewed workload) — re-reads and
re-verifies (checksum, trailer-strip) that same block every time, with
nothing keeping recently-used blocks resident. D8 already specifies the
key shape and ownership model; this phase had to decide the eviction
policy, what specifically gets cached, and how caching integrates with
`sstable::Reader` without disrupting every existing (pre-Phase-5,
no-cache) call site.

## Decision — LRU, not clock

FR7 says "LRU/clock-style," leaving the specific policy open. This
project uses a classic doubly-linked-list-plus-hash-map LRU
(`cache::BlockCache`, `include/pebbledb/cache/block_cache.hpp`,
`src/cache/block_cache.cpp`): `std::list<std::pair<Key, std::string>>`
ordered most-recently-used-first, with an `std::unordered_map<Key,
list::iterator>` index for O(1) lookup. `Lookup()`'s hit path
`splice()`s the found node to the front in O(1) (no copy, no
reallocation, and the index's iterator stays valid — `std::list::splice`
never invalidates iterators). Clock (an approximate-LRU
second-chance-bit scheme over a fixed-size circular buffer) exists
specifically to avoid LRU's per-access list-splice cost under heavy
concurrent contention — a real concern for a production buffer pool, but
not one this project's scope needs to solve: true LRU is simpler to
implement correctly, simpler to reason about and test (eviction order is
exactly, deterministically "least recently used," not "approximately"),
and this codebase has no evidence yet (no Phase 9 benchmark data) that
lock contention on this cache is actually a bottleneck worth clock's
added complexity to avoid. If a future benchmark does show contention
here, swapping the eviction policy is an internal change behind this
same public API — no caller (`sstable::Reader`) needs to change.

## Decision — key shape and ownership: exactly D8's default

`BlockCache::Key = {file_id: uint64, block_offset: uint64}`. `file_id`
is the SSTable's file number (`DB::FlushLocked()`/`Recover()` pass a
table's `manifest::TableFileInfo::file_number` straight through to
`sstable::Reader::Open()`'s `file_id` parameter) — reusing the file
numbering scheme ADR 0013 already established rather than inventing a
second table-identifier space. `block_offset` is the data block's byte
offset within that file (already unique within one table, since data
blocks are non-overlapping — ADR 0010). Together they are globally
unique across every table a DB has open, which is exactly why *both*
halves of the key are needed: two different tables' first data blocks
both legitimately live at file offset 0, and without `file_id`
distinguishing them, a cache shared across every open `Reader` (see
below) would serve one table's bytes for a lookup against a completely
different table —
`tests/integration/test_sstable_filter_and_cache.cpp`'s
`SharedCacheAcrossTwoTablesDoesNotCollideOnOffset` constructs exactly
this scenario (two tables, same key text, same first-block offset) and
confirms each `Reader` gets back its own table's value.

"Cache owns block copies" (not a pointer/view into a `Reader`'s own
buffer): `Insert()` takes its `std::string block` argument by value and
stores it directly; a `Lookup()` hit returns an owned copy out to the
caller. This is the only safe option given the cache's lifetime is
independent of any single `Reader`'s or any single `Get()` call's
lifetime (see below) — a view into memory some other object might free
or overwrite would not be sound to share this way.

## Decision — one cache instance shared across every `sstable::Reader` a `DB` opens

`DB` owns exactly one `cache::BlockCache` (`block_cache_`, sized by
`Options::block_cache_capacity_bytes`, default 8 MiB), created once in
the persistent-mode constructor and passed to every `sstable::Reader::Open()`
call `DB::Recover()`/`FlushLocked()` make. This is what makes the
`(file_id, block_offset)` key shape necessary in the first place (a
cache privately owned by one `Reader` would only ever need
`block_offset` as its key) — and is the natural, useful shape for an LSM
engine's cache: a hot key that happens to move from one SSTable to
another across a flush/compaction still benefits from whatever locality
the cache already captured, and the *total* memory budget for cached
blocks is one number a caller configures once, not
`per-table-capacity × number-of-open-tables`.

## Decision — what gets cached: post-verification data-block content, not raw on-disk bytes, and only data blocks

`sstable::Reader::ReadDataBlock()` (the single call site `Get()` and
`ForEachEntry()` both route every data-block read through) caches the
*already-checksum-verified, trailer-stripped entries bytes* — exactly
what a caller needs to start decoding entries — not the raw on-disk
bytes (entries + compression-type byte + CRC-32C). A cache hit therefore
skips both the `pread` *and* the re-verification step, not just the
`pread`; re-verifying a checksum on every hit against data this process
already verified once and holds under its own exclusive ownership (no
other writer can ever mutate a `Reader`'s already-cached bytes — SSTables
are immutable, invariant 2) would be pure waste.

Only *data* blocks go through this cache. The index block and the filter
block (Phase 5, ADR 0014) are each loaded once, eagerly, at
`Reader::Open()`, and held directly in that `Reader`'s own memory for its
entire lifetime — there is nothing repeated to cache for either of them
(each is read from disk exactly once per `Reader` regardless of how many
`Get()` calls follow), so routing them through the shared cache would
only add bookkeeping overhead for no benefit.

## Decision — `Insert()` never rejects on size; the cache's only real invariant is "never exceed capacity_bytes after any call returns"

A block larger than the entire cache's capacity is still inserted
(`Insert()` has no "too big, refuse it" special case) — but the same
unconditional "evict least-recently-used until back at or under
capacity" trim step that runs after every `Insert()` then evicts it
again immediately, since it alone cannot fit. The net, observable effect
is that inserting an oversized block leaves the cache empty rather than
"caching" something it structurally cannot hold — a simpler contract
than either rejecting oversized inserts outright (an extra branch, and a
silent behavior difference a caller would need to know about) or
special-casing "protect the just-inserted entry from immediate
self-eviction" (which would break the cache's own capacity invariant for
exactly the pathological case that invariant matters most for). `capacity_bytes
== 0` is a legal, degenerate configuration under this same rule — every
`Insert()` immediately evicts itself, every `Lookup()` misses — useful as
an explicit "caching off" control for an apples-to-apples comparison
against a real cache without `sstable::Reader` needing a separate
`cache_ == nullptr` code path to test both configurations.
(`tests/unit/test_block_cache.cpp`'s `OversizedSingleBlockIsInsertedThenImmediatelyEvicted`
and `ZeroCapacityCacheAlwaysMisses` cover both cases directly.)

## Decision — thread safety: one internal `std::mutex`, no exposure to `sstable::Reader`'s own concurrency story

Every `BlockCache` public method takes an internal `std::mutex` for its
entire body. This is deliberately simple (not sharded, not lock-free):
`sstable::Reader::Get()` is already safe for concurrent invocation (its
own state is read-only after `Open()`, plus `std::atomic` counters for
`filter_checks()`/`cache_hits()`/etc.), and multiple concurrent
`DB::Get()` calls only ever take `DB`'s *shared* lock (ADR 0003) — so
this cache genuinely can be hit concurrently from multiple reader
threads and must not itself introduce a data race, but nothing about
this project's Phase 4–6 scope (spec Part 4 rule 3) calls for a
finer-grained locking scheme here either.
`tests/unit/test_block_cache.cpp`'s `ConcurrentLookupAndInsertStayMemorySafe`
is the same style of TSan-validated concurrency smoke test
`tests/unit/test_db_concurrency.cpp` already established for `DB` itself
— no assertions on exact hit/miss counts (the threads race by design),
only on the absence of undefined behavior.

## Consequences

- `tests/integration/test_sstable_filter_and_cache.cpp`'s
  `CacheAvoidsRepeatedDataBlockReadsForSameKey` (paired with its control,
  `WithoutCacheEveryLookupReadsADataBlock`) is this phase's direct,
  deterministic evidence for the "cache effects" half of the roadmap
  exit criterion — 20 repeated lookups of the same key cause exactly one
  real `data_blocks_read()`, not 20.
- Benchmark numbers (Phase 9) that vary `block_cache_capacity_bytes`
  must report which capacity was used — the same discipline ADR 0006
  established for WAL sync policy and ADR 0009 for SSTable block size:
  cache capacity measurably trades memory for read amplification, and
  comparing results across different capacities without labeling would
  be equally misleading.
- A future Phase 6 (compaction), which removes SSTable files from the
  active set, will need to call `BlockCache::Erase(file_id)` for each
  removed table so stale cached blocks can never be served after the
  file itself is gone — that method already exists on this class's
  public API (unused by anything in Phase 4/5) specifically so Phase 6
  does not need to modify `BlockCache` itself to use it.
