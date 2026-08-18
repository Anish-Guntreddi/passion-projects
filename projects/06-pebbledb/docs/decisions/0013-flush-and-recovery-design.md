# ADR 0013: Flush and recovery design — wiring WAL/MemTable/SSTable/manifest into `DB`

- **Status:** Accepted
- **Phase:** 4 (Flush + manifest)
- **Spec requirement:** roadmap Phase 4 — "Safe immutable-table
  publication, manifest update, recovered-WAL clearing per design."
  Exit criterion: "DB survives restart with data migrated WAL/MemTable
  -> SSTable." Invariant 5 (spec §1.5): "Manifest never intentionally
  references a partially published table."
- **Related:** ADR 0012 (manifest format/atomicity), ADR 0003
  (concurrency model), ADR 0006 (WAL sync policy), ADR 0011 (SSTable
  writer file-creation semantics).

## Context

Phases 0–3 built four independently-tested pieces — `Status`/`Slice`
(ADR 0002), the WAL (ADR 0005–0007), `MemTable`/`MemTableList` (ADR
0008), and `sstable::Writer`/`Reader` (ADR 0009–0011) — but `DB` itself
remained exactly Phase 0's in-memory `std::map`, with no path to disk at
all. This phase had to answer: how does `DB` actually use those four
pieces together, and specifically, four sub-questions:

1. **Does `DB`'s in-memory-only mode disappear, or stay?** Every
   Phase 0–3 unit/concurrency test (`tests/unit/test_db_map_reference.cpp`,
   `tests/unit/test_db_concurrency.cpp`) exercises `DB db;` — a
   default-constructed, no-path, no-persistence instance — as a
   deliberate "put/get/delete semantics tested without persistence"
   reference model (Phase 0's own exit criterion). Making persistence
   *mandatory* would either break those tests' premise or force them to
   route through a real filesystem for something they were designed
   specifically not to need.
2. **File numbering and naming.** WAL segments and SSTables both need
   on-disk names; the manifest needs to track which WAL, if any, must be
   replayed, and which tables currently constitute the DB.
3. **When does a flush happen?** Spec Part 4 rule 3 gates background
   workers behind "the synchronous engine is correct and recoverable
   first" — Phase 7 owns background flush/compaction, not this phase.
4. **What, exactly, does "recovered-WAL clearing" mean**, and how does
   `DB::Open()` reconstruct the right in-memory state from whatever
   combination of manifest + WAL + SSTables it finds on disk?

## Decision — `DB` keeps two supported, equally real modes

`DB()` (default-constructed) remains a fully supported in-memory-only
mode: no WAL, no SSTables, no manifest, no persistence across a restart
— backed by `memtable::MemTableList` alone (not the old raw `std::map` —
see the next decision). `DB::Open(dbname, &db)` is the new, real
persistent mode wired exactly per `docs/architecture.md`'s write/read
path diagram. `DB::persistent()` reports which mode a given instance is
in; every method branches on it explicitly rather than one mode being a
"lesser" implementation of the other.

This is a deliberate choice, not a compromise forced by not wanting to
touch existing tests: an ephemeral, zero-filesystem-dependency store is
a legitimate, useful configuration in its own right (scratch state,
testing, an oracle for property tests), and every one of this project's
own Phase 0-era tests already documents exactly that use case. Making
`Open()`'s directory argument mandatory would not make the *persistent*
engine any more correct — it would only delete a valid, already-tested
configuration. `Flush()`/`Compact()` are `OK` no-ops in this mode (there
is nothing on disk to act on); `FlushOnEmptyMemtableIsOkNoOp`-style
semantics carry over from Phase 0 unchanged.

## Decision — `DB`'s internals are unified around `MemTableList`, not a second parallel data structure

Both modes route `Put`/`Delete`/`Get` through the exact same
`memtable::MemTableList` (`memtables_`) and the exact same merge logic
in `DB::Get()` — the in-memory mode simply never has anything in
`tables_` to fall through to. This replaces Phase 0's separate
`std::map<std::string, std::string>` entirely (there is no second,
parallel in-memory representation to keep in sync). `memtable_list.hpp`'s
own doc comments already anticipated this ("expects its owner (DB, once
wired in at Phase 4) to apply ADR 0003's... discipline") — this decision
makes that literal rather than aspirational, and means the in-memory
mode is now a *regression test* for `MemTableList`'s correctness (via
`tests/unit/test_db_map_reference.cpp`'s randomized-operations-vs-oracle
property test) rather than a separate, never-exercised code path.

## Decision — file numbering: one shared, monotonic counter; fixed-name manifest; numbered WAL/SSTable files

`util::WalFileName`/`SstableFileName` (`include/pebbledb/util/filenames.hpp`)
format a single `std::uint64_t` file number as a zero-padded 6-digit
name with a format-identifying extension (`000007.wal`, `000007.sst`).
`DB::next_file_number_` is the one shared counter both WAL segments and
SSTables draw from (never reused across a restart — it is itself
persisted as `manifest::ManifestData::next_file_number`, and every
`SaveManifest()` call records its current value *before* any file
allocated against it could be silently reused). The manifest itself uses
a fixed name (`MANIFEST`, ADR 0012), deliberately outside this numbered
scheme — there is exactly one manifest per DB directory at all times.

## Decision — flush is synchronous and foreground in Phases 4–6

`DB::Flush()`/the auto-triggered flush inside `Put()`/`Delete()` (when
`memtables_.ActiveShouldFlush()` crosses `Options::memtable_size_threshold_bytes`)
run to completion, holding `mutex_` exclusively for the flush's entire
duration, inside whatever call triggered them. This is explicitly a
Phase 4–6 simplification, not a permanent design: spec Part 4 rule 3
("no complex concurrency... before the synchronous engine is correct and
recoverable — Phases 0–6 gate Phase 7") and the roadmap's own Phase 7
deliverable ("Flush/compaction off foreground path") both anticipate
exactly this ADR making flush synchronous now and a later ADR moving it
to a background worker once the synchronous version is proven correct.
A `Put()`/`Delete()` call that happens to cross the threshold pays the
full cost of a flush (an SSTable write + a manifest save) before
returning — an accepted latency spike this phase does not attempt to
hide, and something Phase 9's benchmark study will need to measure and
label explicitly (spec §1.9's sync-policy-labeling discipline, extended
to "was this write's latency inflated by a synchronous flush").

## Decision — `DB::FlushLocked()`'s algorithm and invariant-5 ordering

> **Amended by [ADR 0016](0016-flush-retry-and-recovery-crash-safety-fixes.md):** the
> single-oldest-immutable / freeze-only-if-ImmutableCount==0 algorithm
> described below is the original Phase 4 design; ADR 0016 amends it
> with flush-retry and recovery crash-safety fixes. ADR 0016 is the current algorithm.

1. If nothing is frozen yet (`memtables_.ImmutableCount() == 0`): if the
   active MemTable is also empty, return `OK` (nothing pending — a
   deliberate no-op, not an error). Otherwise `Freeze()` it.
2. Operate on the **oldest** unpublished immutable MemTable
   (`memtables_.immutables().back()`) — not "whichever one `Freeze()`
   just produced." This distinction matters for **retries**: if an
   earlier `Flush()` call already froze a MemTable and then failed
   partway through (the SSTable write or the manifest publish itself
   failed), `memtables_.active()` is empty again by the time a caller
   retries (`Freeze()` already swapped in a fresh one) — checking
   `active().empty()` on that retry would wrongly conclude "nothing
   pending" and silently strand the still-unpublished frozen table
   forever. Resuming from `immutables().back()` instead means a retry
   always makes progress on whatever is genuinely still unpublished.
   (This was caught by `tests/recovery/test_manifest_recovery.cpp`'s
   `FlushFailureAtSstableCreationLeavesDataFullyRecoverable` during
   development — an earlier version of this method checked
   `active().empty()` unconditionally and failed exactly this way on
   retry; the test's assertion that a retried `Flush()` actually
   produces an SSTable caught it immediately.)
3. Allocate a file number, `sstable::Writer::Open()` + `Add()` every
   entry (values *and* tombstones — a flush must not drop delete markers,
   invariant 4) + `Finish()`. Any failure here returns immediately: the
   frozen MemTable is untouched (still fully readable via `Get()`'s
   merge, still fully present in the still-intact current WAL), so
   nothing is lost — the caller simply retries later.
4. Only **after** `Finish()` returns `OK` (the table is durably complete
   on disk — invariant 2) does this method touch anything about
   *publishing* it: open a `sstable::Reader` on the new file, allocate
   and create a brand-new WAL segment, build the new `ManifestData`
   (existing tables + the new one, plus the new WAL/next-file-number),
   and call `manifest::SaveManifest()`. **This ordering is invariant
   5's entire enforcement mechanism**: the manifest is never asked to
   reference a table until that table's own durability is already
   established; if `SaveManifest()` itself then fails, the SSTable and
   the new (empty) WAL both exist on disk but are referenced by
   nothing — harmless orphans (a future Phase 6 compaction could reclaim
   the SSTable; the empty WAL is simply never opened again), not a
   correctness problem, and not a status this method hides: it is still
   returned to the caller as a failed `Flush()`.
5. Only after `SaveManifest()` returns `OK` — the actual publish
   point — does this method adopt the new state in memory (`tables_`,
   `memtables_.RemoveImmutable()`, swap in the new `wal_writer_`) and
   delete the now-superseded old WAL file (best-effort; a failed
   deletion leaves a harmless orphan, not a correctness problem, so it
   is not itself propagated as a `Flush()` failure). This delete is the
   roadmap's "recovered-WAL clearing" deliverable, literally: the old
   WAL's data is now durably captured in the SSTable just published, so
   replaying it again on a future `Open()` would be redundant (and, if
   the WAL were somehow *not* deleted and *were* replayed again, would
   not be a correctness bug either — MemTable/`MemTableList` Put/Delete
   are idempotent-per-final-value — but deleting it is still both the
   literal roadmap deliverable and good disk hygiene).

## Decision — `DB::Recover()`'s algorithm

Called once, from `Open()`, before a persistent `DB` is handed to its
caller:

1. `manifest::LoadManifest()`. `NotFound` means a brand-new directory —
   proceed with `ManifestData`'s defaults (`next_file_number = 1`,
   `wal_file_number = 0`, no tables). Any other non-`OK` status (a
   manifest exists but fails to decode) is fatal — `Open()` fails rather
   than guessing (ADR 0012's `LoadManifest` doc comment explains why
   auto-recovering from a corrupted manifest is out of scope).
2. Open an `sstable::Reader` (sharing `block_cache_`, keyed by file
   number — D8/ADR 0015) for every table the manifest lists, in the
   manifest's own stored order (oldest-flushed-first — `tables_[i]`
   always corresponds to `manifest_.tables[i]`).
3. If `manifest_.wal_file_number != 0`: replay that WAL
   (`wal::Reader::Replay`) directly into the (currently empty) active
   MemTable, tracking the highest sequence number seen. `kEndOfLog` and
   `kTruncated` (a torn trailing record — the expected shape of an
   unclean shutdown, `docs/durability.md`) both mean recovery completed
   normally. `kCorruption`/`kUnsupportedVersion` (bytes that were fully,
   successfully written are corrupted) are treated as fatal — consistent
   with the WAL's own no-resynchronization policy (ADR 0006) generalized
   one layer up, rather than silently discarding whatever followed the
   corruption point without telling the caller. Then reopen that same
   WAL file for append (`wal::Writer::Open`), so new writes continue
   landing in the segment recovery just proved is intact.
4. Otherwise (a genuinely fresh DB — no manifest existed): allocate a
   brand-new WAL file number, create it, and **immediately**
   `SaveManifest()` a manifest recording it — before `Open()` returns
   `OK` to the caller. This closes the same "allocated-but-unrecorded
   file number could be silently reused after a crash" gap ADR 0012's
   `MANIFEST.tmp` discipline closes for manifest *updates*, applied here
   to the very first file this project's own code ever allocates in a
   fresh directory. (`tests/recovery/test_manifest_recovery.cpp`'s
   `FreshOpenPersistsManifestBeforeReturning` asserts this directly.)
5. `next_sequence_` is set from the highest sequence number observed
   across the manifest and the replay (`max(manifest.last_sequence,
   every replayed record's sequence) + 1`), so newly issued sequence
   numbers never collide with anything already durable.
6. `live_key_count_` (`DB::Stats`) is seeded once here via a full
   merge-scan across every recovered layer (`ComputeLiveKeyCount()`) —
   see the stats-tracking decision below for why this is a one-time cost
   here, not a per-`GetStats()`-call one.

## Decision — `live_key_count` is tracked incrementally, not recomputed per `GetStats()` call

An earlier version of this phase computed `Stats::live_key_count` via a
full merge-scan across every MemTable and SSTable **on every
`GetStats()` call**. This is correct but was measurably too expensive:
`tests/unit/test_db_concurrency.cpp`'s
`ConcurrentReadersDuringWritesStayMemorySafe` (a pre-existing Phase 0
test — four reader threads calling `GetStats()` in a tight loop
alongside a writer doing 2000 `Put()`s) **timed out under ASan** with
that design, because each of the many thousands of `GetStats()` calls
those reader threads make was independently allocating a
`std::map`-based scan across the whole in-memory state.

The fix: `DB::live_key_count_` is a plain member, seeded once (via that
same full-scan `ComputeLiveKeyCount()`) in `Recover()`, then kept correct
incrementally by `Put()`/`Delete()` via
`IsKeyCurrentlyLiveLocked()` — one merged-read-path lookup (the same
active/immutable-MemTables-then-SSTables-newest-first merge `Get()`
itself performs) consulted *before* applying the write, to decide
whether the key's live/not-live status is actually changing. `GetStats()`
itself is now `O(1)`. A flush does not need to touch `live_key_count_`
at all — moving already-visible data from a MemTable to an SSTable does
not change *which* keys are currently live, only where they are stored.

This adds one extra lookup per `Put()`/`Delete()` in exchange for making
every `GetStats()` call cheap — the right trade for a stats field that
existing tests (and any real caller polling stats, e.g. for a dashboard
or a benchmark harness) call far more often, and in tighter loops, than
they call `Put`/`Delete`. A lower-layer read failure during this
bookkeeping lookup (e.g. a corrupted SSTable block) is treated as
"not live" rather than propagated — `live_key_count` is documented as a
best-effort stat, and must never block or fail the write it is
bookkeeping for; a real `Get()` call still surfaces that same corruption
as an error normally (invariant 7 is not weakened by this).

## Consequences

- `tests/integration/test_db_persistence.cpp` is this phase's direct
  exit-criterion evidence: real restarts (`db.reset()` then a fresh
  `DB::Open()` on the same directory — no special shutdown step, the
  same "simulated crash" pattern `tests/recovery` already uses for the
  WAL alone) after WAL-only writes, after an explicit flush, after
  several automatic threshold-triggered flushes, and a
  randomized-operations-vs-`std::map`-oracle property test that restarts
  every few hundred operations (spec §1.8's "across restarts" property
  test, previously not possible before persistence existed).
- `tests/recovery/test_manifest_recovery.cpp` fault-injects failures at
  each of `FlushLocked()`'s two invariant-5-relevant boundaries (SSTable
  creation, manifest publish) and confirms no data is ever lost and a
  fresh `Open()` recovers cleanly either way.
- `docs/architecture.md`'s "What's implemented so far" and `README.md`
  are updated in this same change to describe `DB` as wired, per spec
  Part 4 rule 2's documentation-in-the-same-change discipline (applied
  here to the write/read-path wiring, not just a binary format).
- Phase 6 (compaction) and Phase 7 (background workers) both build on
  this ADR's shape directly: compaction is "produce a new `ManifestData`
  with some tables removed and others added, publish it the same way";
  background flush is "move `FlushLocked()`'s body onto a worker thread,
  behind the same bounded-queue coordination Phase 7 designs," neither
  of which requires revisiting the algorithm described here.
