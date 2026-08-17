# ADR-0003: Stats interface is a syscall, not a procfs-like VFS

**Status:** Accepted (spec default); implementation deferred to Phase 2
**Decision:** D3 (spec §1.10)
**Date:** 2026-08-17 (recorded ahead of implementation, per the spec's
kickoff prompt: "use the §1.10 defaults for D3, D4, D6, D7 with ADRs"
before implementing phase-by-phase)

## Context

FR3 requires a kernel/user statistics interface. D3 is procfs-like VFS
vs. a dedicated stats syscall; the spec's own recommended default is
"stats syscall for MVP; procfs stretch."

## Decision

Adopt the spec default: a dedicated syscall-based statistics interface
for Phase 2 (process accounting) and Phase 3 (`xvtop`), not a
procfs-like virtual filesystem. Not implemented in Phase 0/1 (no
accounting fields exist yet); recorded now so Phase 1's tracing
syscall and Phase 2's stats syscall are designed against one
consistent "small syscalls, safe scope, no new VFS machinery"
philosophy.

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

`xvtop` (Phase 3) will poll a stats syscall in a loop rather than
open/read files under `/proc`. If procfs is later attempted as a
stretch goal, it can be layered on top of the same underlying
per-process struct without changing the struct's shape.
