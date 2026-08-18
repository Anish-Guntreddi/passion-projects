# ADR 0016: Flush-retry and recovery crash-safety fixes

- **Status:** Accepted
- **Phase:** 4 (Flush + manifest) — post-review hardening, not a new
  feature
- **Spec requirement:** invariant 1 (spec §1.5: "an acknowledged durable
  write is recoverable after process crash per the documented sync
  policy"); invariant 5 ("manifest never intentionally references a
  partially published table"); roadmap Phase 4 exit criterion ("DB
  survives restart with data migrated WAL/MemTable -> SSTable").
- **Related:** ADR 0012 (manifest format/atomicity), ADR 0013 (flush and
  recovery design — this ADR amends its `FlushLocked()`/`Recover()`
  algorithms directly), ADR 0006 (WAL sync policy and resync policy).

## Context

An external review of Phase 4/5's implementation (Codex) found three
related, real crash-safety bugs in `DB::FlushLocked()`,
`DB::Recover()`, and `manifest::SaveManifest()` — all three were
independently reproduced against the actual built library before this
ADR's fixes, not merely reasoned about statically. All three share one
root theme: a piece of code correctly reasoned about the *first* failure
it was designed to survive, but not about *composition* — a failure
followed by more writes, or by a second failure, or by a partial
success.

### Bug 1 — a flush retry could lose writes acknowledged in between

`FlushLocked()`'s original algorithm (ADR 0013) resumed a retry from
`memtables_.immutables().back()` — the single oldest still-frozen
immutable MemTable — specifically *to* fix an earlier bug where checking
`active().empty()` on a retry wrongly concluded "nothing pending" (see
ADR 0013's own account of that fix). But resuming from just the oldest
immutable has the same shape of blind spot one level up: it does not
account for anything that accumulated in the *active* MemTable between
the failed attempt and the retry. Because the WAL was — and, per fix
\#1 below, still is — only rotated to a new segment *after* a publish
succeeds, those intervening writes were appended into the very same WAL
segment the eventually-successful retry then deleted as "superseded."

Reproduced directly: `Put(k1)`, `Put(k2)` → `Flush()` fails (a blocked
`MANIFEST.tmp` path, mirroring the project's existing fault-injection
tests) → `Put(k3)` (acknowledged, `Put()` returns `OK`) → unblock →
`Flush()` succeeds (flushing only `k1`/`k2`) → the old WAL, which still
held `k3`'s only durable copy, is deleted → simulated crash → reopen →
`Get(k3)` returns `NotFound`. None of the 231 tests that existed before
this ADR exercised "writes made between a failed flush and a successful
retry," so this shipped untested.

### Bug 2 — a torn WAL tail, reopened for append without truncation, could brick `Open()` on a second crash

A torn trailing WAL record (`wal::ReadResult::kTruncated`) is, by
design (ADR 0006, `docs/durability.md`), the *expected* shape of an
unclean shutdown's last, in-flight write, and `Recover()` correctly
treats it as non-fatal — everything before it replays normally.  But
`Recover()` then reopened that exact file for append
(`wal::Writer::Open()` → `util::PosixFile::OpenAppend()`) without first
discarding those untrusted torn bytes. New records appended after that
point landed immediately after the old torn ones, byte-for-byte, with
nothing marking the boundary.

Reproduced directly: `Put(k1)`, `Put(k2)` → crash → truncate the WAL's
last byte (a torn record) → reopen #1 succeeds, `k1` recovered, `k2`
correctly dropped as torn → `Put(k3)` (acknowledged, appended
immediately after the torn bytes) → crash → reopen #2 fails outright
with a corruption error, because replay walks the earlier valid records
fine, then hits the torn-bytes-followed-immediately-by-a-new-record
region and cannot parse it as anything but a single corrupted record
(the WAL's own no-resynchronization policy, ADR 0006, correctly refuses
to guess). This does not just lose `k3` — it makes the *entire* DB
directory permanently unopenable, taking previously-durable `k1` down
with it. `tests/corruption/test_manifest_corruption.cpp`'s existing torn
-tail test only verified a *single* reopen, never a second write-then
-crash cycle after it, so this shipped untested too.

### Bug 3 — a post-rename directory-fsync failure was indistinguishable from "nothing happened"

`manifest::SaveManifest()` (D7's write-new + fsync + atomic rename +
directory-fsync, ADR 0012) returned a single `Status` covering all four
steps. If `rename()` (step 3 — the actual publish point) succeeded but
the trailing directory `fsync()` (step 4) then failed, the function
still returned non-`OK` — identical to a failure *before* the rename
ever happened. `DB::FlushLocked()` treated any non-`OK` `SaveManifest()`
return as "not published" and kept operating on its old in-memory
`manifest_`/`wal_writer_`/`tables_` — but the on-disk `MANIFEST` file
already pointed at the new state. Any further writes the live process
made would keep landing in a WAL its own manifest file no longer
referenced; a later crash's `Recover()` (which trusts whatever is
durably on disk, and does not care whether the directory-fsync that
followed the rename which produced it succeeded) would then silently
diverge from what the live process had been doing — the same class of
lost-acknowledged-write bug as \#1 and \#2, via a third door. This
specific interior timing (rename succeeds, directory-fsync then fails)
is harder to trigger than \#1/\#2 and was not fault-injected — see
"Consequences" below for why — but the code-path reasoning is
unambiguous, and the fix is cheap and structural.

## Decision

**Fix 1 — `FlushLocked()` freezes unconditionally and drains every
pending immutable before ever touching the WAL.** Every call now
freezes whatever is currently active (if non-empty) — not gated behind
`ImmutableCount() == 0` — so writes accumulated across any number of
prior failed attempts are folded in, potentially producing more than one
pending immutable MemTable. A `while (ImmutableCount() > 0)` loop then
flushes each one, oldest first, to its own SSTable + its own manifest
publish (still leaving the WAL segment untouched — `wal_file_number` in
each of these interim manifests still names the *current*, unrotated
WAL). Only once the loop drains to zero — meaning every write that was
ever logged into that WAL segment is now durably captured in a published
SSTable, guaranteed because `mutex_` is held exclusively for the
call's entire duration so nothing new can be appended in between
iterations — does `FlushLocked()` rotate to a new WAL (its own, final
manifest publish) and delete the old one. This preserves the original
design's core safety property ("never rotate/delete a WAL until
everything it might contain is proven durable elsewhere") while
extending it to cover *retries with intervening writes*, not just a
single freeze-then-flush cycle.

An alternative considered and rejected: rotate the WAL immediately at
freeze time (before the SSTable write), so new writes after a freeze
target a fresh WAL from the start. This was rejected because it opens a
*different* crash window: a crash between that early rotation and the
eventual manifest publish would leave writes made in between recorded
only in a WAL the manifest does not yet reference (an orphan file
`Recover()` never looks at), reproducing the same class of loss through
a fourth door. Deferring rotation to the very end — after everything
pending is proven durable — has no such window: whatever a crash catches
mid-flush, the still-unrotated, still-intact old WAL still has every
record needed to reconstruct the correct state, exactly as in the
original design.

**Fix 2 — truncate a torn WAL tail before ever reopening it for
append.** `wal::Reader` now tracks `bytes_consumed_`, exposed as
`valid_prefix_length()`: the file offset immediately after the last
successfully-decoded record. `Recover()`, whenever a WAL's replay
terminal is `kTruncated`, calls the new `util::TruncateFile(path,
valid_prefix_length())` — `ftruncate()` + `fsync()` on a writable fd
opened specifically for this — *before* reopening that same path via
`wal::Writer::Open()`/`OpenAppend()`. This is a plain content mutation
of an already-existing file (shrinking it), not a directory-entry
creation or replacement, so — unlike `SyncDirectory()` — no separate
containing-directory fsync applies (see `util/posix_file.hpp`'s
`SyncDirectory()` doc comment for that distinction). If the truncation
itself is not yet durable across a *literal* power loss (out of this
project's testable scope regardless, see `docs/durability.md`'s existing
boundary), the worst case is simply replaying and discarding the same
torn tail again on a subsequent crash — never corruption, since no new
write can have landed past it until the truncation (or an equivalent
effect from a genuinely lost update) has actually taken hold.

**Fix 3 — `SaveManifest()` reports whether the publish point was
actually reached.** `SaveManifest()` gained an optional `bool*
published` out-param, set unconditionally before any I/O and then set
to `true` the instant `rename()` succeeds — regardless of what the
trailing `SyncDirectory()` call does afterward. Every call site in
`DB` that previously treated "non-`OK` return" as "nothing changed" now
checks `published` first: `false` really does mean nothing changed (safe
to return the error as-is, exactly as before); `true` means the on-disk
manifest already is the new content, so the in-memory state
(`manifest_`, `tables_`, `memtables_`, `wal_writer_`/`current_wal_number_`
as applicable) is adopted immediately, and only *then* is the
(still-real, still-reported) error returned to the caller as a degraded
-but-safe outcome. `DB::Recover()`'s fresh-DB-path call to
`SaveManifest()` deliberately keeps the plain (no `published`) call —
see the inline comment at that call site — because that path has no
"stale in-memory state to keep operating on" risk: any failure there
fails `Open()` outright, and a retried `Open()` naturally re-derives the
correct state from whatever is actually on disk either way.

## Consequences

- `tests/recovery/test_manifest_recovery.cpp` gains
  `FlushRetryWithInterveningWritesLosesNoAcknowledgedData` (bug 1, driven
  end-to-end through the same fault-injection style the file's other
  tests already use) and `SaveManifestPublishedIsFalseWhenRenameNeverHappens`
  / `SaveManifestPublishedIsTrueOnFullSuccess` (bug 3's `published`
  contract, both outcomes).
- `tests/corruption/test_manifest_corruption.cpp` gains
  `SecondUncleanShutdownAfterATornTailStillOpensCleanlyAndRecoversNewWrites`
  (bug 2, a real two-crash sequence).
- Bug 3's specific "rename succeeds, directory-fsync then fails" interior
  timing is not fault-injected: there is no portable, deterministic way
  to force a real directory `fsync()` to fail from inside a hosted,
  non-root test (the same boundary `docs/durability.md`'s "What this
  test suite verifies, and what it cannot" section already draws around
  real kernel/power-loss behavior generally). The `published` contract
  itself, and `FlushLocked()`'s handling of both of its possible values,
  is fully covered instead — which is what actually determines
  correctness at the call sites that matter.
- All 235 tests (231 pre-existing + 4 new) pass across every project
  build variant: Debug-none, Debug-asan+ubsan, Debug-ubsan-only,
  Debug-tsan, and Release-none.
- No public API shape changed except `manifest::SaveManifest()` gaining
  a defaulted (`= nullptr`) trailing parameter — every existing call site
  outside `db.cpp` (there are none) continues to compile and behave
  identically.
- `DB::Stats::flush_count` now increments once per SSTable a
  `FlushLocked()` call actually publishes, rather than once per call —
  identical to the old behavior for the common case (one pending
  immutable per call) and more accurate, not less, when a retry drains
  more than one in a single call.
