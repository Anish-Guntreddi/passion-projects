# ADR 0006: WAL sync policy and resync-on-corruption policy (spec decision D3)

- **Status:** Accepted
- **Phase:** 1 (WAL)
- **Spec decision:** D3 — "WAL sync policy" — default per spec §1.10:
  **configurable {every-write, batched, explicit}; benchmarks label the
  mode.**

## Context

Spec invariant 1 (§1.5) is "an acknowledged durable write is recoverable
after process crash **per the documented sync policy**" — durability is
explicitly relative to a chosen policy, not an unconditional promise, and
§1.9 requires benchmarks to "never compare results across different
durability settings without labeling." That means the policy needs to be a
first-class, explicit choice per `Writer`, and "acknowledged" needs a precise
definition per policy so later benchmark reports (Phase 9) can label results
correctly.

Separately, `wal::Reader` needs a policy for what to do when it encounters a
record that fails validation mid-log: try to scan forward and find the next
plausible-looking record ("resynchronize"), or stop there and trust nothing
after that point. This is a distinct decision from the sync policy but is
covered by the same ADR because both concern "what does the WAL guarantee
after a crash," and both are implemented in the Phase 1 WAL module.

## Decision — sync policy (`wal::SyncPolicy`)

Three modes, chosen at `Writer::Open` time and fixed for that `Writer`'s
lifetime:

- **`kEveryWrite`** — `AddRecord()` does not return `OK` until `fsync()` has
  succeeded for that record. "Acknowledged" for this policy means exactly
  "this specific record's bytes are durable" — the strongest and slowest
  option, appropriate for correctness-focused tests (used throughout
  `tests/recovery` and `tests/corruption`) and for benchmarks explicitly
  measuring per-write fsync cost (spec §1.9).
- **`kBatched`** — `fsync()` runs every `batch_size` records (checked `>= 1`
  at `Open` time; `Open` returns `InvalidArgument` for `batch_size == 0`),
  or whenever `Sync()` is called explicitly. "Acknowledged" for an
  individual `AddRecord()` call under this policy means "written to the
  file's in-kernel buffer," **not** "durable" — durability for a given
  record is only guaranteed once a subsequent sync (automatic, at the batch
  boundary, or explicit) has completed. This is the throughput/durability
  middle ground the spec's default anticipates.
- **`kExplicit`** — `AddRecord()` never fsyncs automatically; only an
  explicit `Sync()` call does. "Acknowledged" (`AddRecord` returning `OK`)
  under this policy means only "handed to the OS," with no durability claim
  at all until `Sync()` returns `OK`. Existing purely for benchmarking the
  write-side cost floor with fsync removed from the critical path
  entirely — `tests/recovery/test_wal_truncation_recovery.cpp`'s
  `UnsyncedExplicitWritesAreStillVisibleToASubsequentReader` test
  deliberately documents the boundary of what a hosted unit test can verify
  here: bytes written without an intervening `Sync()` are still visible to
  a subsequent `Reader` in-process (because nothing simulated real power
  loss or a dropped OS page cache), which is a statement about this test
  environment, not a durability guarantee about `kExplicit` itself.
- **A record is only "acknowledged" once `AddRecord()`/`Sync()` actually
  returns `OK`.** If encoding or validating a record fails before it is
  appended, `AddRecord()` returns a non-OK `Status` and nothing is written —
  see "record-size validation" below, which is the specific gap this ADR's
  policy closes.
- **Every benchmark report (Phase 9) must state which `SyncPolicy` was in
  effect** alongside its numbers, per spec §1.9's "never compare... without
  labeling" rule; this is a reporting discipline, not something the code
  enforces mechanically.

## Decision — record-size validation happens before acknowledgment, not after

`EncodeRecord()`'s header fields (`key_length`, `value_length`) are
`uint32_t`, but the WAL format's own sanity bounds
(`wal::kMaxKeyLength` = 1 MiB, `wal::kMaxValueLength` = 64 MiB —
`include/pebbledb/wal/record.hpp`) are stricter than what fits in a
`uint32_t`, and are exactly the bounds `DecodeHeader()` enforces on
**replay**. A `Writer` that encoded and fsynced a record exceeding those
bounds would therefore acknowledge a write (`AddRecord()` returns `OK`) that
`Reader::ReadNext()` can never successfully decode: `DecodeHeader()` reports
`kBadLength`, which `Reader` maps to `ReadResult::kCorruption`, which halts
replay at that record and discards it and everything after it. That is a
direct violation of invariant 1 ("an acknowledged... write is recoverable")
— an oversized record must never be allowed to reach the point of
acknowledgment.

**Decision:** `Writer::AddRecord()` validates `record.key.size()` and (for
`kPut`) `record.value.size()` against `kMaxKeyLength`/`kMaxValueLength`
*before* encoding or writing anything, and returns
`Status::InvalidArgument(...)` without touching the file if either bound is
exceeded. This makes "oversized" a rejected write (never acknowledged,
never partially written), not a silently-lost one. The check lives in the
`Writer`, not in `EncodeRecord()` itself, because `EncodeRecord()` is also
used directly by tests that intentionally construct hand-crafted
boundary/corruption cases (`tests/unit/test_wal_record_codec.cpp`) where an
assertion at encode time would be the wrong layer — `Writer` is the only
call site that makes a durability *acknowledgment*, so it is the only place
this check needs to gate.

## Decision — resync-on-corruption policy (`wal::Reader`)

- **No resynchronization past a bad record.** The WAL is read strictly
  front-to-back; the first record that is not fully present and valid
  (`kTruncated`, `kCorruption`, or `kUnsupportedVersion`) ends replay
  immediately. `Reader` does not scan forward looking for the next
  byte offset that happens to decode as a plausible header.
- **Why:** PebbleDB's WAL is single-writer and strictly append-only (ADR
  0003, ADR 0004) — there is exactly one legitimate way well-formed bytes
  can appear in this file, in this order, written by this process. A
  corrupted or truncated record already means either (a) an unclean
  shutdown truncated the last, in-flight write (the expected, benign case —
  reported as `kTruncated`), or (b) something is wrong with the file that a
  single-writer, single-process WAL should never produce on its own (bit
  rot, external truncation/corruption, disk fault — reported as
  `kCorruption`). In case (b), any bytes after the corruption point were
  written *after* whatever corrupted the file, so treating them as trusted
  data by "resynchronizing" to the next plausible-looking header risks
  replaying attacker- or fault-controlled bytes as if they were legitimate
  application writes — silently returning garbage, which is exactly what
  invariant 7 (§1.5) prohibits ("checksums detect corrupted/truncated
  records rather than silently returning garbage"). Stopping at the first
  bad record is the conservative, correct choice for this WAL's threat and
  failure model.
- **Once `Reader::ReadNext()` returns a non-`kOk` result, the reader is
  permanently exhausted**: every subsequent call returns that same terminal
  result again without attempting to read further bytes
  (`tests/corruption/test_wal_corruption.cpp`'s
  `ReaderStaysExhaustedAfterFirstCorruptionOnRepeatedCalls`).

## Consequences

- `AddRecord()` gains one more way to fail (`InvalidArgument` for an
  oversized key/value) that callers must handle; this is a compile-visible,
  synchronous rejection at the write call site, which is strictly easier to
  handle correctly than a write that appeared to succeed and then vanished
  silently on the next restart.
- Because resynchronization never happens, a single corrupted or truncated
  record anywhere in the WAL caps recovery at "everything strictly before
  it" — there is no partial-credit recovery of later, structurally-valid-
  looking records past a corruption point. This is a deliberate
  conservative trade-off (see above), not an oversight; a future format
  version could add per-segment checksums or periodic resync markers if a
  different trade-off is ever needed, but that would be a new ADR and a new
  format version, not a change to v1's semantics.
- Benchmark numbers (Phase 9) for write latency must be reported per
  `SyncPolicy` and never averaged or compared across policies without that
  label attached.
