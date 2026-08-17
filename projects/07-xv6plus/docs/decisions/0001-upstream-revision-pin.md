# ADR-0001: Pin upstream to mit-pdos/xv6-riscv @ xv6-riscv-rev5

**Status:** Accepted (implementer decision, see Context)
**Decision:** D1 (spec §1.10)
**Date:** 2026-08-17

## Context

The spec marks D1 ("Upstream/course revision to pin") as a human
decision, to be recorded in Phase 0. This project was built by a
non-interactive implementer agent (Auto Mode, no human available to
name a specific university course/semester revision mid-build). Per
the portfolio orchestration's own decision log ("adopt each spec's
recommended default; conservative... where specs say 'human
decision'"), the implementer picked the most defensible, objective
default rather than blocking on an unavailable human input.

## Decision

Pin `kernel/`, `user/`, `mkfs/`, `Makefile`, `README`,
`LICENSE.upstream-xv6`, `.gdbinit.tmpl-riscv`, `.editorconfig`, and
`.dir-locals.el` to:

- **Repo:** `mit-pdos/xv6-riscv`
- **Tag:** `xv6-riscv-rev5`
- **Commit:** `7d7adbb1b0acbd67c9766a20d0f9900fef2789fa`
- **Fetched:** 2026-08-17

Vendored verbatim (copied, not a git submodule -- this project lives
inside a portfolio monorepo owned by an external orchestrator; see
ADR-0008). All xv6-plus modifications are additive diffs on top of
this exact tree, enumerated in `docs/upstream-delta.md`.

## Rationale

- `mit-pdos/xv6-riscv` is the canonical, actively maintained RISC-V
  xv6 base used by MIT 6.1810, the course this spec explicitly targets
  (§1.1: "MIT's xv6 RISC-V teaching kernel").
- A tag marks a maintainer-curated release boundary, which is a more
  reproducible, more stable base than pinning today's tip-of-main
  commit (the upstream repo is under active development -- `git log`
  showed same-day commits at fetch time).
- The pinned tree builds and boots cleanly with this environment's
  toolchain (`riscv64-unknown-elf-gcc` 13.2, `qemu-system-riscv64`
  8.2) -- verified as the Phase 0 exit criterion.

## Alternatives considered

- **Pin `main` HEAD.** Rejected: no curated release boundary, more
  exposure to not-yet-widely-tested in-flight changes.
- **Pin an older tag.** Rejected: no stated reason (course syllabus,
  known-good semester) to prefer a stale tag over the latest one.
- **Use the `xv6-labs-*` per-lab-branch teaching repo instead.**
  Rejected: that repo's branches pre-apply lab solutions, which is in
  direct tension with the §1.2 honesty requirement and with this
  project's D2 default of starting from zero incorporated labs (see
  ADR-0002).

## Consequences

If a human reviewer later specifies a different required course
revision, re-pinning means re-vendoring `kernel/`, `user/`, `mkfs/`
from that revision and re-diffing `docs/upstream-delta.md`. The
xv6-plus-original changes through Phase 1 are small and localized
(one new field, one new syscall, two new user programs, all called
out with `// xv6-plus:` comments), so re-basing onto a different pin
is expected to be low-risk.
