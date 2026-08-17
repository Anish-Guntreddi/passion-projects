# ADR 0007: CUB Comparisons (D7)

- **Status:** Accepted (deferred)
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D7 — default: stretch.

## Decision
CUB-based reference/ceiling comparisons (analogous to the cuBLAS ceiling
used for GEMM, FR2) are **out of scope for MVP** and treated as a stretch
goal, per the spec default. No CUB headers are included and no CUB
comparison numbers are collected in Phases 0–7 unless explicitly revisited
as a stretch item after MVP acceptance criteria (spec §1.7) are met.

## Consequences
- Reduction/scan ladders (Phase 2) are evaluated only against each other
  and against the CPU reference, not against `cub::DeviceReduce` /
  `cub::DeviceScan`. This keeps the ladder's lesson focused on what the
  developer's own kernel changes bought, per hard constraint 2 (never
  copy an optimized kernel wholesale from external source) — CUB is a
  library dependency, not a technique to reimplement.
