# ADR-0003: Stats interface is a syscall, not a procfs-like VFS

**Status:** Accepted; implemented in Phase 2 (validated by `xvtop` in
Phase 3)
**Decision:** D3 (spec §1.10)
**Date:** 2026-08-17 (recorded ahead of implementation, per the spec's
kickoff prompt: "use the §1.10 defaults for D3, D4, D6, D7 with ADRs"
before implementing phase-by-phase; revisited after Phase 2/3
implementation)

## Context

FR3 requires a kernel/user statistics interface. D3 is procfs-like VFS
vs. a dedicated stats syscall; the spec's own recommended default is
"stats syscall for MVP; procfs stretch."

## Decision

Adopt the spec default: a dedicated syscall-based statistics interface
for Phase 2 (process accounting) and Phase 3 (`xvtop`), not a
procfs-like virtual filesystem. Implemented in Phase 2 as `xvstat(2)`
(`kernel/sysproc.c: sys_xvstat`, `kernel/proc.c: procstat()`,
`kernel/pstat.h: struct xv_pstat`) -- see `docs/accounting.md` for the
full design -- and consumed in Phase 3 by `xvtop` polling it in a
loop, exactly as predicted below. Recorded ahead of implementation so
Phase 1's tracing syscall and Phase 2's stats syscall were designed
against one consistent "small syscalls, safe scope, no new VFS
machinery" philosophy.

## Rationale

xv6's file/inode/VFS layer (`kernel/fs.c`, `kernel/file.c`) is a
substantial subsystem with its own lock ordering and lifecycle rules.
A synthetic `/proc`-like filesystem is a much larger surface (new
inode type, new `read()` dispatch path, careful interaction with
existing directory/inode invariants) than the MVP needs, and risks
invariant #6 (VM/user-kernel isolation) for a Phase-2-scoped feature.
A syscall that `copyout()`s a small, fixed-layout struct (or array of
structs) to a user buffer follows the same pattern already used
successfully by `kwait()`'s `xstate` copyout, and stays easy to keep
within "safe scope" (validated by `copyout()` itself).

## Consequences

`xvtop` (Phase 3) polls `xvstat(2)` in a loop (`idx = 0..NPROC-1`,
stopping at the first out-of-range index) rather than open/read files
under `/proc` -- see `docs/xvtop.md`. If procfs is later attempted as
a stretch goal, it can be layered on top of the same underlying
`struct xv_pstat` without changing its shape.
