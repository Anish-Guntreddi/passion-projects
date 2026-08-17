# ADR 0003: Concurrency model (spec decision D9)

- **Status:** Accepted
- **Phase:** 0 (library foundation) — revisited at Phase 7 (background
  workers)
- **Spec decision:** D9 — "Concurrency model / multiple writers in MVP" —
  default per spec §1.10: **single-writer, multi-reader**.

## Context

`DB` is used from multiple threads even before any background work exists
(Phase 7): a caller may reasonably issue concurrent `Get()` calls from
several reader threads while a single writer thread issues `Put`/`Delete`
calls. Phase 0's `DB` needs a concurrency contract precise enough to (a) make
that pattern memory-safe today, and (b) state clearly what it does *not*
promise, so Phase 7's background flush/compaction workers have an
unambiguous baseline to extend rather than a contract that has to be
reverse-engineered from the implementation.

Spec Part 4 rule 3 gates this: "No complex concurrency ... before the
synchronous engine is correct and recoverable (Phases 0–6 gate Phase 7)." So
Phase 0's answer is deliberately the simplest model that is still genuinely
safe under concurrent foreground use, not a preview of Phase 7's
coordination.

## Decision

- **Single-writer, multi-reader**, enforced with one `std::shared_mutex`
  (`DB::mutex_`):
  - `Put` and `Delete` take the lock **exclusively** (`std::unique_lock`):
    at most one write is in flight at a time, and no read overlaps a write.
  - `Get` and `GetStats` take the lock **shared** (`std::shared_lock`):
    any number of reads may proceed concurrently with each other, just not
    concurrently with a write.
  - This is exactly the mutual-exclusion contract `std::map` (the Phase 0
    backing store) needs: concurrent reads of a `std::map` are safe,
    concurrent read+write or write+write are not.
- **Per-call counters that a *shared*-locked method mutates must be
  independently atomic**, not merely "protected" by the shared lock. A
  `shared_lock` only guarantees no *exclusive* lock holder overlaps it — it
  does not serialize multiple concurrent shared-lock holders against each
  other. `Get()`'s hit/miss/call counters (`get_count`, `get_hit_count`,
  `get_miss_count`) are exactly this case: two threads calling `Get()`
  simultaneously both legitimately hold the shared lock at once, so a plain
  `++stats_.get_count` from each is a data race (lost updates, and under the
  formal C++ memory model, undefined behavior) even though both calls are
  individually "correct" with respect to the map itself. `DB` stores these
  three counters as `std::atomic<std::uint64_t>` for exactly this reason;
  `GetStats()` folds their relaxed loads into the returned `Stats` snapshot.
  `put_count`, `delete_count`, and `live_key_count` do not need this
  treatment: they are only ever written while holding the lock
  *exclusively* (inside `Put`/`Delete`), so at most one thread touches them
  at a time and the shared_mutex's own exclusion is sufficient.
- **What this contract does *not* claim:**
  - It says nothing about atomicity *across* multiple calls — e.g. a
    `Get()` immediately followed by a `Put()` from the same logical
    operation is two independent critical sections, not one transaction.
    PebbleDB has no cross-key or multi-call transactions (spec §1.6
    non-goals).
  - It is not a statement about concurrent *background* work. Flush and
    compaction (Phase 7) introduce a second class of "writer" (a background
    thread mutating on-disk state) that this ADR's model does not yet
    coordinate with; that coordination is designed and tested at Phase 7,
    gated by Phases 0–6 being correct and recoverable first (spec Part 4
    rule 3).

## Consequences

- Concurrent `Put`/`Get`/`Delete`/`GetStats` calls from multiple threads are
  memory-safe today and validated under ThreadSanitizer
  (`build/Debug-tsan`, `PEBBLEDB_ENABLE_TSAN`) —
  `tests/unit/test_db_concurrency.cpp` runs concurrent `Get()` calls
  specifically to exercise the shared-lock/atomic-counter interaction this
  ADR describes; without the atomic counters, that test fails under TSan
  with a data race report on `stats_`.
- Read throughput scales with concurrent readers (no reader ever blocks
  another reader); write throughput does not (writes fully serialize),
  which matches an LSM engine's usual shape — the WAL append path (Phase 1
  onward) is the true write bottleneck, not this in-memory lock.
- `util::PosixFile::OpenAppend` opens with `O_APPEND` specifically so that
  even if this single-writer assumption were ever relaxed, appends stay
  atomically positioned at end-of-file — not a scenario PebbleDB relies on
  today, but the correct primitive regardless (see
  `include/pebbledb/util/posix_file.hpp`).
- Phase 7 must either extend this single mutex's scope to cover background
  mutation of on-disk state, or introduce a separate, explicitly-documented
  coordination mechanism (bounded queues, a compaction lock, etc.) — that
  design is out of scope for this ADR and is written up when Phase 7 starts.
