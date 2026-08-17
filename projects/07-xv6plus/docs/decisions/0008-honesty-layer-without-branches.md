# ADR-0008: Honesty layer without a branch-per-category structure

**Status:** Accepted (constraint-driven adaptation, not a spec open decision)
**Date:** 2026-08-17

## Context

The Tech Stack Plan's VCS row calls for "Original-project branch
structure over pinned upstream" as the mechanism for the §1.2 honesty
requirement: clearly distinguishing (A) upstream xv6 code, (B)
completed educational lab modifications, (C) original xv6-plus
extensions.

This project is built inside a portfolio monorepo (all nine projects
under `projects/<nn-name>/`, per the portfolio orchestration's
decision log) whose git history -- branches included -- is owned
entirely by an external orchestrator. The implementer building this
project is explicitly barred from running any git commands at all.

## Decision

Replace the branch-per-category mechanism with three things that
don't require touching git:

1. **A pinned, explicitly-cited upstream revision** (ADR-0001), with
   the vendored tree copied in unmodified except where diffed.
2. **`docs/upstream-delta.md`**, enumerating every touched and every
   added file, categorized (A)/(B)/(C) per §1.2, kept current every
   phase.
3. **Inline `// xv6-plus:` comments at every original change site**
   (e.g. `kernel/proc.h`'s `trace_mask` field, `kernel/syscall.c`'s
   `SYS_trace` entries, the whole of `user/xtrace.c` and
   `user/tracetest.c`) that say outright this is xv6-plus-original and
   point at `docs/upstream-delta.md` -- so the distinction is visible
   even to a reader who never opens `docs/`.

## Rationale

A branch per category is a fine mechanism when the implementer owns
commit history directly, which is the brief's default assumption of a
standalone repository. It does not compose with "the orchestrator owns
all git operations" in a shared monorepo: an implementer creating
xv6-plus-specific branches inside someone else's repository would
itself be exactly the kind of out-of-scope git action the portfolio's
hard rules prohibit. The delta-doc-plus-inline-comment combination
achieves the same reviewability goal -- a reader can tell upstream
from original at a glance, at either the file level or the line level
-- without needing branch access.

## Consequences

`docs/upstream-delta.md` becomes load-bearing and must be kept current
every phase, not just Phase 0/1 -- a stale delta doc would silently
defeat the honesty requirement this ADR exists to preserve. If this
project is ever split out of the monorepo into its own standalone
repository, the originally-recommended branch structure (a branch or
tag for the pristine pin, plus feature branches per phase) can be
reconstructed retroactively from this same delta doc.
