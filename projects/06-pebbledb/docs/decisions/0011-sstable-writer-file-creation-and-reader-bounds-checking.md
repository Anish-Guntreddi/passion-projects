# ADR 0011: SSTable writer file-creation semantics and reader block-handle bounds checking

- **Status:** Accepted
- **Phase:** 3 (SSTable) — post-hoc hardening found during Phase 2/3 review
- **Related:** invariant 2 ("SSTable contents are immutable after
  successful creation"), invariant 7 ("checksums detect
  corrupted/truncated records rather than silently returning garbage")

## Context

Phase 3 review turned up two related gaps, both in how `sstable::Writer`
and `sstable::Reader` trust file-position/size values instead of
independently verifying them:

1. `sstable::Writer::Open()` originally reused
   `util::PosixFile::OpenAppend` (`O_CREAT | O_WRONLY | O_APPEND`, no
   truncation, no rejection of an existing path) — the same primitive
   `wal::Writer` correctly uses, since reopening an existing WAL file to
   keep appending to it on restart is exactly the WAL's intended
   behavior. `sstable::Writer` has no such "resume" concept: it always
   starts `file_offset_` at 0 and records every `BlockHandle` it writes
   relative to that. If `Open()` were ever called against a path that
   already had bytes in it — a stale partial file left by a prior failed
   `Finish()`, a retried flush, a reused/collided file number — the new
   table's recorded offsets would describe where bytes *should* be if
   the file were empty, while `O_APPEND` actually lands them after the
   pre-existing content. This was reproduced directly: opening a second
   `Writer` on a path a first, already-`Finish()`ed `Writer` had used
   produced a file where the second table's index still pointed at
   offset 0 — physically the *first* table's leftover data. Reads against
   the second table then returned either a ghost value that was never
   written to it, or `kNotFound` for a key that genuinely was — silently
   wrong answers, not errors, which is exactly what invariant 2 and
   invariant 7 exist to rule out.
2. Neither `Reader::Open()` (for the footer's `index_handle`/
   `filter_handle`) nor `Reader::ReadBlock()` (for each index entry's
   data-block handle) checked a `BlockHandle`'s `offset + size` against
   the file's actual size before calling `util::PosixFile::ReadAt`, which
   immediately does `out->assign(n, '\0')` — an eager allocation of `n`
   bytes — before any I/O happens. A hand-crafted footer with a correctly
   computed checksum (so `DecodeFooter` accepts it) but an absurd
   `index_handle.size` therefore crashed the process with an uncaught
   `std::bad_alloc` instead of returning `Status::Corruption`.
   `format.hpp`'s `DecodeEntry`/`DecodeIndexEntry` deliberately skip a
   declared-length sanity bound (see `DecodeEntry`'s own comment) on the
   reasoning that a block is always read as one fixed-size, already-
   bounded buffer before any entry inside it is decoded — but that
   reasoning applies to lengths *within* an already-loaded block, not to
   the `BlockHandle`s that decide how large a buffer to load in the first
   place. Those handles come from the footer and from index entries, are
   covered by a checksum only at the block/footer level (not individually
   sanity-bounded), and directly drive the read/allocation size — so they
   needed their own check, independent of `format.hpp`'s entry-decode
   layer.

## Decision — `Writer::Open()` requires a path with no real content (`util::PosixFile::OpenNew`)

`util::PosixFile` gained `OpenNew()`: it `stat()`s `path` first (mirroring
`OpenAppend()`'s own existed-before check), and fails with
`Status::IOError` if `path` exists **and its size is non-zero** — a path
that is absent, or present but already empty (0 bytes), is accepted
either way and always yields a file that starts writing at offset 0
(`O_CREAT | O_WRONLY | O_TRUNC`; `O_TRUNC` is defense-in-depth for the
already-accepted stat-then-open race window below, not the primary
guarantee — the `stat()` check is). No `O_APPEND` either: unlike the WAL,
which documents `O_APPEND` as correct specifically because multiple fds
could in principle append concurrently (ADR 0003), exactly one `Writer`
ever holds this fd, so plain sequential `write()` calls from a freshly
opened fd already land at the offsets `Writer` expects.

**Why "no real content" and not a strict "path must not exist" (`O_EXCL`)
check:** the test suite's `TempFile` helper (`tests/support/temp_file.hpp`)
deliberately pre-creates an empty placeholder file via `mkstemp` to
reserve a race-free path, and dozens of existing SSTable `Writer` tests
construct one and pass its `.path()` straight to `Writer::Open()`. A
zero-byte file is not "existing content" in any sense that matters to the
actual bug this ADR fixes (mismatched offsets from *real*, non-zero
bytes already on disk) — it is indistinguishable from "does not exist
yet" for every purpose `Writer::Open()` cares about. Rejecting it too
would have forced a sweep of every SSTable test call site onto a
different path-reservation helper for no correctness benefit; accepting it
keeps the fix scoped exactly to the bug (silently writing into/being
offset by *real* pre-existing bytes) without changing established test
infrastructure. If this call creates `path`'s directory entry (it did not
already exist), the containing directory is fsync'd before returning
`OK`, mirroring `OpenAppend()`'s own rule (`docs/durability.md`, "File
creation durability: directory entries").

`wal::Writer::Open()` is unchanged and still uses `OpenAppend()` — the
"reopen an existing file and keep appending to it after restart" case is
correct and required there.

## Decision — `sstable::Reader` bounds-checks every `BlockHandle` against the file's real size before reading it

`Reader` now records the file's size (`file_size_`, set once in `Open()`
from the same `fstat()`-backed `Size()` call it already made) and adds a
single helper, `ValidateHandleInBounds(handle, file_size, what)`, checked
as `handle.size > file_size || handle.offset > file_size - handle.size`
(the subtraction form specifically to avoid `offset + size` itself
overflowing `std::uint64_t` and wrapping past the check). This is called
in exactly two places, both before any `pread`/allocation happens for
that handle:

- `Open()`, for the footer's `index_handle` **and** `filter_handle` —
  `filter_handle` is never actually read in this version (always
  zero-length; ADR 0009), but a corrupted footer could still declare a
  bogus non-zero one, and rejecting that at `Open()` is cheap and
  consistent with treating footer corruption as an `Open()`-time concern
  (ADR 0010's "index block: ... loaded eagerly" rationale).
- `ReadBlock()` itself, which every data-block read (`Get()`,
  `ForEachEntry()`, each via an index entry's handle) and the index-block
  read (`Open()`, via the footer's handle) all go through — one check
  site covers every handle the `Reader` ever dereferences, rather than
  duplicating it at each call site.

An out-of-range handle is reported as `Status::Corruption` (same
category as every other structural-validation failure in this file),
never allowed to reach `util::PosixFile::ReadAt`.

## Consequences

- A `Writer::Open()` call against a path with real (non-empty) content is
  now a hard error instead of a latent data-corruption bug; any future
  caller that flushes MemTables to numbered SSTable files (Phase 4) must
  treat file-number allocation as "always a fresh number," which was
  already the intended design (spec FR3's per-table files) — this ADR
  just makes a violation of that fail loudly at the point of the mistake
  instead of producing a silently wrong table.
- `sstable::Reader::Open()` and every subsequent block read now have a
  hard, file-size-derived ceiling on how large a single read/allocation a
  corrupted file can trigger — closing the same class of "declared-length
  drives an unbounded read" risk that `wal::DecodeHeader`'s
  `kMaxKeyLength`/`kMaxValueLength` close for the WAL, just enforced at
  the `Reader` layer (against real file size) rather than the
  `format.hpp` decode layer, since that is where the relevant ground
  truth (actual bytes on disk) is available.
- Regression coverage:
  `tests/integration/test_sstable_writer_reader.cpp`'s
  `Invariant2_WriterOpenOnExistingFileFailsInsteadOfCorruptingIt`
  reproduces the original two-`Writer`s-one-path scenario end to end;
  `tests/corruption/test_sstable_corruption.cpp`'s
  `Invariant7_OversizedFooterIndexHandleIsRejectedNotReadAsHugeAllocation`,
  `Invariant7_OversizedFooterFilterHandleIsRejectedAtOpen`, and
  `Invariant7_OversizedIndexEntryBlockHandleIsRejectedNotReadAsHugeAllocation`
  exercise the bounds-check fix at both the footer and per-index-entry
  level.
