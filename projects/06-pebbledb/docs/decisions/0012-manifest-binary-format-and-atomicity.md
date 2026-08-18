# ADR 0012: Manifest v1 binary format and atomicity (spec decision D7)

- **Status:** Accepted
- **Phase:** 4 (Flush + manifest)
- **Spec requirement:** FR5 — "Manifest/version metadata sufficient to
  recover the active file set; crash-safe updates per the stated
  durability model." FR10 — "All binary formats versioned from the
  beginning."
- **Spec decision:** D7 — "Manifest atomicity approach" — default per
  spec §1.10: **write-new + fsync + atomic rename.**

## Context

Once SSTable flush exists (this phase's other deliverable — ADR 0013),
`DB` needs a durable record of "which SSTable files currently constitute
this database's on-disk state, and which WAL segment (if any) still
needs replaying on top of them." Invariant 5 (spec §1.5) requires this
record never intentionally reference a table that was not fully and
successfully written — a torn or half-written manifest update would be
exactly as dangerous as a torn SSTable.

Two format questions and one durability question needed answers before
any flush code could be written:

1. Binary layout — should this reuse the WAL/SSTable formats'
   established conventions, or does a manifest's different shape (one
   small, whole-file record instead of an append-only log or a
   many-independent-blocks file) call for something different?
2. Versioned file names or one fixed name? (LevelDB's actual design uses
   `MANIFEST-000001`-style versioned files plus a separate `CURRENT`
   file pointing at the active one.)
3. How is an update made crash-safe, given the spec's D7 default already
   picks the *mechanism* (write-new + fsync + atomic rename) but not the
   specifics of applying it here?

## Decision — reuse the WAL/SSTable formats' byte-level conventions

Every multi-byte field is fixed-width, explicitly little-endian
(`util::PutFixed32`/`PutFixed64`), and the whole file is covered by one
CRC-32C checksum — the same conventions ADR 0007 established for the WAL
and ADR 0010 reused for the SSTable format, reused again here rather than
re-derived a third time. `docs/file-format.md`'s "Manifest format"
section is the field-by-field source of truth; this ADR explains why,
not what.

Layout (`include/pebbledb/manifest/format.hpp`,
`src/manifest/format.cpp`): a fixed 37-byte header (`checksum(4) +
magic(4) + format_version(1) + next_file_number(8) + wal_file_number(8)
+ last_sequence(8) + num_tables(4)`) followed by `num_tables` fixed
12-byte table entries (`file_number(8) + level(4)`). `magic =
0x50424D46` ("PBMF" read as big-endian bytes, the same hex-dump-mnemonic
convention as the WAL's "PBK1" and the SSTable's "PBST").

## Decision — one manifest file at a fixed name, not versioned files + `CURRENT`

`manifest::kManifestFileName` ("MANIFEST") is the *only* on-disk name
this project's manifest ever uses — there is no `MANIFEST-NNNNNN`
sequence and no separate `CURRENT` indirection file naming which one is
active. LevelDB's versioned-manifest design exists to support
*incremental* updates (appending small "edit" records to the current
manifest file, only writing a whole new snapshot occasionally) — a
legitimate optimization, but one this project's scope does not need:
`DB::FlushLocked()` (ADR 0013) calls `SaveManifest()` at most once per
flush, and a flush is already a bounded, infrequent, foreground
operation (not a hot path this format needs to optimize incremental
writes for). Writing the *entire* small manifest (37 bytes + 12 bytes
per table — even a thousand tables is ~12 KB) fresh on every update, and
always reading it whole, is simpler to implement, simpler to reason
about ("there is exactly one manifest file; if it exists and decodes, it
is the whole truth"), and simpler to test — at the cost of doing
asymptotically more I/O per update than an incremental-edit-log design
would, which is an acceptable trade at this project's scale (spec Part
3's non-goals already rule out "complex... matrices" elsewhere in this
codebase in the same spirit).

## Decision — write-new + fsync + atomic rename (D7), via a disposable, reused-by-name temp file

`manifest::SaveManifest()` (`include/pebbledb/manifest/manifest.hpp`,
`src/manifest/manifest.cpp`) implements D7 literally:

1. Encode the new `ManifestData` and write it to
   `dbname/MANIFEST.tmp` (`kManifestTempFileName`) via
   `util::PosixFile::OpenTruncate()` — a new primitive added in this
   phase specifically for this use: unlike `OpenNew()` (SSTable writer,
   ADR 0011 — rejects a path with real existing content) or
   `OpenAppend()` (WAL writer — resumes an existing file), a
   write-new-manifest temp file is *always* meant to be clobbered by
   whatever the next `SaveManifest()` call writes, including a stale
   leftover temp file abandoned by a prior crashed `SaveManifest()` call
   — there is nothing worth preserving in an abandoned temp file (the
   real manifest at `MANIFEST` was never touched until the rename step
   below, so an abandoned temp file is simply garbage to discard, not
   data to recover).
2. `fsync()` that file, so its bytes are durable *before* anything
   references it by its final name.
3. `rename(MANIFEST.tmp, MANIFEST)` — atomic on the same filesystem
   (POSIX guarantees a concurrent opener of `MANIFEST` sees either the
   complete old file or the complete new one, never a mix). This is the
   actual publish point.
4. `fsync()` the containing directory — `util::SyncDirectory()`, the
   same primitive `PosixFile::OpenAppend()`/`OpenNew()`/`OpenTruncate()`
   already use internally for "this call just created a new directory
   entry," exposed publicly for this call site because a *rename*
   mutates a directory entry the same way a *create* does (what the name
   `MANIFEST` points to changed), and needs the identical durability
   treatment — `docs/durability.md`'s "File creation durability:
   directory entries" section, generalized from creation to replacement.

`manifest::LoadManifest()` is the read side: `stat()`s for existence
first (returning `Status::NotFound()` — not an error — for "no manifest
yet," which `DB::Recover()`, ADR 0013, treats as "start fresh"), reads
the whole file, and calls `format::DecodeManifest()`, mapping any
non-`kOk` `DecodeStatus` to `Status::Corruption` uniformly (unlike the
WAL/SSTable readers, the manifest format has no "unsupported-but-
recognized" content this project's writer can produce, so there is no
`Status::NotSupported` case to distinguish here).

## Decision — validation order handles the "checksum coverage length itself depends on a field" bootstrapping problem

`DecodeManifest()`'s declared `num_tables` is not yet checksum-verified
when it must first be used to compute the buffer's *expected* total size
— the same problem the WAL's `key_length`/`value_length` already solve
(ADR 0006) at a smaller scale. The validation order (magic → format
version → `num_tables` against `kMaxTableCount` → buffer size against
the now-computed expected size → checksum last) mirrors that established
pattern: a corrupted `num_tables` field is bounds-checked and rejected
*before* it is ever used to size an allocation or a checksum-coverage
computation, closing the same "declared length drives unbounded
work" class of risk ADR 0006/ADR 0011 close elsewhere in this codebase
(invariant 7). `kMaxTableCount` (2²⁰, ~1,048,576) is a generous sanity
bound, not a claim about a realistic table count.

Unlike the WAL (an appendable log, where trailing bytes past a decoded
record are simply "not yet read") or an SSTable (independently-addressed
blocks), the manifest is a **whole-file** format: `DecodeManifest()`
rejects a buffer *longer* than `num_tables` implies
(`DecodeStatus::kBadLength`) exactly as strictly as one that is too
short (`kTruncated`) — there is no legitimate reason for extra trailing
bytes to exist in a file this project only ever produces via
`EncodeManifest()`'s exact output.

## Consequences

- `tests/unit/test_manifest_format.cpp` covers pure in-memory encode/
  decode round-trips (including table order being preserved, not
  sorted — `TableOrderIsPreservedNotSorted`, matching the
  oldest-flushed-first convention ADR 0013 relies on) and every
  `DecodeStatus` enumerator, including the "corrupted `num_tables` must
  be rejected before the checksum is even consulted" case
  (`DecodeRejectsTableCountExceedingSanityBound`).
- `tests/corruption/test_manifest_corruption.cpp` corrupts real on-disk
  manifest bytes (checksum, truncation) and confirms `DB::Open()` fails
  cleanly with `Status::Corruption` rather than crashing or silently
  discarding the table list (`OpenFailsCleanlyWhenManifestChecksumIsCorrupted`,
  `OpenFailsCleanlyWhenManifestIsTruncated`).
- `tests/recovery/test_manifest_recovery.cpp` fault-injects a failed
  `SaveManifest()` call (blocking `MANIFEST.tmp`'s path with a directory)
  and confirms invariant 5 holds end-to-end: the SSTable that would have
  been published is left an orphan, referenced by nothing, and every key
  remains recoverable via the still-intact old WAL
  (`FlushFailureAtManifestPublishLeavesTableOrphanedButDataRecoverable`).
- A future Phase 6 (compaction) that needs to *remove* table entries
  (not just append them) can do so with this exact same `SaveManifest()`
  call — it always writes the complete, current `ManifestData`, so
  removing an entry from the in-memory `tables` vector before calling it
  requires no format or atomicity change here.
