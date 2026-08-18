# Durability

This document is the authoritative contract for what "acknowledged" and
"durable" mean in PebbleDB's WAL, and what this project's test suite can and
cannot actually verify about that contract. See ADR 0006 for the design
rationale behind the decisions summarized here.

## Spec invariant 1

> An acknowledged durable write is recoverable after process crash per the
> documented sync policy. (spec §1.5, invariant 1)

Durability is explicitly **relative to a chosen sync policy** — this
document exists so "acknowledged" has one precise meaning per policy, not an
implied unconditional one.

## Sync policies (`wal::SyncPolicy`)

| Policy | When `fsync()` runs | What `AddRecord()` returning `OK` means |
|---|---|---|
| `kEveryWrite` | After every single `AddRecord()` call, before it returns | This record's bytes are durable on stable storage right now. |
| `kBatched` | Every `batch_size` records, or on an explicit `Sync()` call | This record is written to the file (visible to a subsequent read in-process), but **not necessarily durable yet** — durability for this specific record is only guaranteed once a sync (automatic batch boundary or explicit) that includes it has completed. |
| `kExplicit` | Only on an explicit `Sync()` call — never automatically | Same as `kBatched`: written, not necessarily durable, until `Sync()` is called and returns `OK`. |

`Writer::sync_count()` / `records_since_sync()` expose exactly when syncs
have happened, so a caller (or a test) can determine, for any given
`AddRecord()` call, whether a subsequent sync has covered it yet.

**A record only becomes "acknowledged" at all once `AddRecord()` (or the
`Sync()` that later covers it) returns `OK`.** If a record fails validation
before ever being written — currently: `record.key.size() >
wal::kMaxKeyLength` or (for `kPut`) `record.value.size() >
wal::kMaxValueLength` — `AddRecord()` returns
`Status::InvalidArgument(...)` and nothing is appended to the file. This
check exists specifically so that "acknowledged" and "recoverable" can never
diverge: an oversized record is rejected at the write call site, not
silently written, fsynced, acknowledged, and then permanently lost on the
next replay (`wal::Reader::ReadNext` cannot successfully decode a record
whose declared length exceeds those same bounds — see "Record-size
validation" below and ADR 0006).

## File creation durability: directory entries

`fsync()` on a file descriptor guarantees the *file's contents* are durable,
but on POSIX filesystems it does **not** by itself guarantee that the
*directory entry* pointing to a newly created file survives a crash — that
requires a separate `fsync()` of the containing directory. Without it, the
following sequence is possible: `wal::Writer::Open()` creates a brand-new
WAL file (`O_CREAT`), `AddRecord()` under `kEveryWrite` appends and fsyncs
the file's contents and returns `OK` (acknowledging the write), the process
crashes, and on restart the directory entry for the file — and therefore the
"acknowledged" record inside it — is gone, because only the file's own
`fsync()` ran, never the directory's.

**Decision:** `util::PosixFile::OpenAppend()` fsyncs the parent directory
once, immediately after creating a new file (i.e. when `O_CREAT` actually
created it, not when the path already existed), before returning success to
the caller. This closes the gap: by the time `Writer::Open()` returns `OK`
for a new WAL path, the directory entry is already durable, so the very
first `AddRecord()`/`Sync()` afterward only needs to make the *file's*
bytes durable to satisfy invariant 1 — the directory half of that guarantee
is already in place. Opening an *existing* path (the normal "reopen a WAL
for append on restart" case) does not re-fsync the directory, since no new
directory entry was created.

## Record-size validation happens before acknowledgment

See ADR 0006 for the full rationale. Summary: `Writer::AddRecord()` checks
`record.key.size()` and (for `kPut`) `record.value.size()` against
`wal::kMaxKeyLength` (1 MiB) / `wal::kMaxValueLength` (64 MiB) *before*
calling `EncodeRecord()` or touching the file. A record exceeding either
bound is rejected with `Status::InvalidArgument(...)` — it is never written,
never fsynced, and therefore never falsely acknowledged. This matters
because those same bounds are what `wal::DecodeHeader()` enforces on
**replay** (`DecodeStatus::kBadLength`, which `Reader` reports as
`ReadResult::kCorruption` and which halts recovery at that record): without
the write-side check, an oversized record could be acknowledged at write
time and then become unrecoverable — and take every record after it in the
log down with it, per the no-resynchronization policy below — the first
time the WAL was replayed.

## Resync-on-corruption policy

`wal::Reader` reads strictly front-to-back and stops at the first record
that is not fully present and valid. It never scans forward past a bad
record looking for the next plausible-looking header ("resynchronization").
Full rationale: ADR 0006. In one line: this WAL is single-writer and
strictly append-only, so there is exactly one legitimate way well-formed
bytes appear in it; anything after a corruption point cannot be trusted to
be an unrelated, later, still-legitimate write, so it is never replayed.

`ReadResult` distinguishes:

- `kTruncated` — a record was started but not fully present (header or
  payload torn). This is the *expected* shape of an unclean shutdown's
  last, in-flight write, and is not treated as an error condition by
  recovery code (everything before it is still replayed normally).
- `kCorruption` — a full record's bytes were present but failed validation
  (bad magic, bad declared length, bad checksum, or bad operation byte).
  This is not the expected shape of a clean crash — it means something
  altered bytes that were already fully written, which a single-writer
  process should not do to its own WAL on an ordinary unclean shutdown.
- `kUnsupportedVersion` — a full header was present with a
  `format_version` newer than this build understands. Reported distinctly
  from `kCorruption` on purpose: an unrecognized future version is not
  evidence of data corruption, and a caller might reasonably want to
  surface it differently (e.g. "upgrade this binary") rather than treating
  it as a bit-rot event.

## What this test suite verifies, and what it cannot

`tests/recovery` and `tests/corruption` write real WAL files through
`wal::Writer`, destroy the `Writer` object with no special shutdown step
(the closest a hosted unit test gets to simulating an abrupt process exit),
truncate or bit-flip real bytes on disk via `tests/support/temp_file.*`
helpers, and confirm `wal::Reader` replays exactly the expected prefix
deterministically. This verifies:

- Replay is byte-for-byte deterministic given whatever is actually on disk.
- Truncation at **every** byte offset (not just record boundaries) is
  handled without ever fabricating a record that wasn't fully and validly
  written (`ReplayIsCorrectAtEveryPossibleTruncationOffset`).
- A corrupted record, and everything after it, is never replayed
  (`Invariant7_*` tests in `tests/corruption`).
- Sync-policy cadence matches its documented contract (`sync_count()`/
  `records_since_sync()` assertions).

It does **not**, and cannot, verify:

- Real kernel/power-loss data loss — that would require dropping the OS
  page cache or cutting power to real hardware, which is unavailable in a
  hosted CI/test environment. "Simulated crash" in this test suite means
  "the `Writer` object is destroyed with no explicit close/finalize step,"
  not "the OS actually lost unflushed writes."
- That `fsync()` itself does what the POSIX spec says it does on every
  possible filesystem/storage stack this code might ever run on. PebbleDB
  trusts the OS/filesystem's `fsync()` contract; verifying that contract is
  out of scope for this project.
- Bytes written under `kExplicit` without an intervening `Sync()` are
  *expected* to still be visible to a subsequent in-process `Reader` in
  these tests (`UnsyncedExplicitWritesAreStillVisibleToASubsequentReader`)
  — that is a statement about this test environment (no real power loss
  occurred), not a durability claim about `kExplicit` itself, which makes
  no durability promise for unsynced records.

## SSTable durability (Phase 3)

`sstable::Writer::Finish()` fsyncs the file once, after the footer has
been written, before returning `OK` — so a `Writer::Finish()` that returns
`OK` means the *entire* table (every data block, the index block, and the
footer) is durable, not just some prefix of it. `Writer::Open()` uses
`util::PosixFile::OpenNew`, not `OpenAppend`'s create-or-reopen semantics
— see ADR 0011 — so creating a brand-new SSTable file gets the same
containing-directory fsync as a brand-new WAL file (see "File creation
durability: directory entries" above) — by the time `Writer::Open()`
returns `OK`, the directory entry is already durable, so `Finish()` only
needs to make the file's own bytes durable. `OpenNew` also guarantees
`Open()` never succeeds against a path that already has *real* content
(a merely-empty placeholder path is fine and always starts a fresh file),
which matters for durability bookkeeping specifically: `Writer`'s
`file_offset_` (and therefore every `BlockHandle` it records) starts from
0 and must exactly match on-disk byte positions, which silently stops
being true if `Open()` ever appended to genuinely pre-existing content
instead of failing outright.

Phase 3's `Writer`/`Reader` are deliberately silent on *when* to call
`Finish()` and then discard the MemTable data that table was flushed
from — `Writer::Finish()` either fully succeeds (whole table durable) or
returns a non-OK `Status` (nothing after that point is guaranteed
durable, and the caller must not treat the table as usable); there is no
partial-success state a caller could observe and mistake for a complete
table. That coordination is Phase 4's `DB::FlushLocked()` — see the next
section.

## Manifest and flush durability (Phase 4)

Full algorithm: ADR 0013. This section states the durability contract
specifically.

**Manifest atomicity (spec decision D7, ADR 0012).**
`manifest::SaveManifest()` writes the new manifest content to a
temporary file, `fsync()`s it, `rename()`s it over the real `MANIFEST`
path (atomic on the same filesystem — a concurrent opener of `MANIFEST`
always sees either the complete old file or the complete new one, never
a mix), then `fsync()`s the containing directory (the rename mutated a
directory entry — "what `MANIFEST` points to" changed — which needs the
exact same directory-fsync treatment as a brand-new file's creation,
above). `SaveManifest()` returning `OK` is the precise point at which
the new manifest content becomes the durable, recoverable truth.

**Invariant 5, mechanically enforced by ordering.** `DB::FlushLocked()`
never calls `SaveManifest()` until the SSTable it is about to reference
has already had `Writer::Finish()` return `OK` — i.e. until that table
is already durably complete on disk (per the SSTable durability
guarantee above). If anything *before* that point fails (the SSTable
write itself), nothing has been published and nothing needs to be
undone — the MemTable being flushed is simply still resident, still
readable, and still backed by its still-intact WAL segment, exactly as
before the failed `Flush()` call. If `SaveManifest()` itself then fails
*after* the SSTable is complete, the table exists on disk but is
referenced by nothing — a harmless orphan, not a correctness problem
(nothing in this project's own code ever opens a table it did not find
listed in a successfully loaded manifest).

**"Recovered-WAL clearing."** Once every currently-pending immutable
MemTable has been published this way — there can be more than one in a
single `FlushLocked()` call, see ADR 0016 below — `DB::FlushLocked()`
rotates to a new WAL and deletes the WAL segment that flush superseded.
This delete happens strictly *after* every one of those publishes, and
is itself best-effort (a failed delete leaves a harmless orphan file —
nothing ever reopens a WAL segment the current manifest does not point
at). This is why a flush failure never loses data: the old WAL is only
ever removed once *everything* it might contain is proven durable
somewhere else first.

**ADR 0016 — three crash-safety fixes found in this phase's original
implementation, post-review.**

1. *A flush retry must account for writes made in between.* An earlier
   version of `FlushLocked()` resumed a retry from only the single
   oldest still-frozen immutable MemTable, not anything that had since
   accumulated in the active MemTable. Because the WAL was only rotated
   *after* a successful publish, a write acknowledged between a failed
   flush attempt and a later successful retry was logged only into the
   same WAL segment that retry's eventual success then deleted as
   "superseded" — silently losing it. Fix: `FlushLocked()` now
   unconditionally freezes whatever is active at the start of every
   call (potentially producing a second, third, ... immutable MemTable
   across repeated failed attempts) and drains *every* pending immutable
   — oldest first, each into its own SSTable + manifest publish — before
   ever rotating or deleting the WAL. Regression test:
   `tests/recovery/test_manifest_recovery.cpp`'s
   `FlushRetryWithInterveningWritesLosesNoAcknowledgedData`.
2. *A torn WAL tail must be truncated before being reopened for append.*
   `Recover()` previously reopened a WAL for append immediately after
   replaying past a torn trailing record (`kTruncated`), without first
   discarding those untrusted bytes. A second unclean shutdown could
   then leave new, valid records concatenated directly after the old
   torn ones, which no later replay can safely tell apart from a single
   corrupted record — permanently failing `Open()` for the whole
   database. Fix: `wal::Reader::valid_prefix_length()` reports how many
   leading bytes of the file were actually trusted (i.e. successfully
   decoded), and `Recover()` calls `util::TruncateFile()` to shrink the
   WAL to exactly that length *before* reopening it for append whenever
   the replay terminal was `kTruncated`. Regression test:
   `tests/corruption/test_manifest_corruption.cpp`'s
   `SecondUncleanShutdownAfterATornTailStillOpensCleanlyAndRecoversNewWrites`.
3. *A post-rename directory-fsync failure is not "nothing happened."*
   `manifest::SaveManifest()`'s failure return did not previously
   distinguish "the `rename()` itself never happened" (the on-disk
   manifest is genuinely unchanged) from "`rename()` succeeded, only the
   trailing directory-fsync afterward failed" (the on-disk manifest
   already *is* the new content). A caller treating both cases as
   "nothing happened" could keep operating on stale in-memory state
   already superseded on disk — the same class of bug as #1, via a third
   door. Fix: `SaveManifest()` takes an optional `bool* published`
   out-param, set to `true` the moment `rename()` succeeds regardless of
   what happens after; `FlushLocked()` (and `Recover()`'s fresh-DB path,
   where the same risk does not apply — see its own inline comment)
   adopts the new manifest state whenever `published` is `true`, even on
   an otherwise-non-OK return, rather than only on a fully-`OK` one.
   Regression tests: `tests/recovery/test_manifest_recovery.cpp`'s
   `SaveManifestPublishedIsFalseWhenRenameNeverHappens` and
   `SaveManifestPublishedIsTrueOnFullSuccess`. The specific "`rename()`
   succeeds but the trailing directory-fsync then fails" interior
   timing is not itself fault-injected (no portable, deterministic way
   to force a directory `fsync()` to fail from inside a hosted test
   without root — the same boundary this document's "What this test
   suite verifies, and what it cannot" section already draws around real
   kernel/power-loss behavior); the `published` contract itself and both
   of its outcomes are.

**Recovery (`DB::Recover()`, called from `DB::Open()`).** A missing
manifest means "fresh DB, start from nothing" (not an error). A manifest
that exists but fails to decode is fatal — `Open()` returns
`Status::Corruption` rather than guessing at a partial or best-effort
recovery (this project's manifest write path is designed so this should
never happen from an ordinary crash — see ADR 0012 — so seeing it means
external corruption, not a workload this project's own crash-safety
covers). The WAL segment the loaded (or freshly created) manifest points
at is replayed exactly per this document's existing WAL recovery
contract above: everything before a torn trailing record (`kTruncated`)
replays normally; a fully-written-but-corrupted record (`kCorruption`)
is fatal to `Open()`, consistent with the WAL's own no-resynchronization
policy (ADR 0006) applied one layer up.

**What this test suite verifies, and what it cannot (Phase 4
extension).** `tests/integration/test_db_persistence.cpp` and
`tests/recovery/test_manifest_recovery.cpp` use the same "destroy the
`DB` object with no special shutdown step, then open a brand-new one on
the same directory" simulated-restart pattern this document's WAL
section already establishes — not a real kernel/power-loss event, for
the same reasons that document already states. `test_manifest_recovery.cpp`
additionally *fault-injects* specific failures (blocking a target path
with a directory so a specific `open()`/`rename()` call fails
deterministically) to exercise the ordering guarantees above directly,
rather than only their happy path.

## Benchmark labeling (spec §1.9)

Every benchmark report that measures write latency or throughput must state
which `SyncPolicy` was in effect. Results across different sync policies
must never be compared or averaged without that label — `kEveryWrite`'s
per-record fsync cost and `kExplicit`'s near-zero write-path cost are not
interchangeable numbers.
