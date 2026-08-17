# ADR 0008: Templating by Dtype (D8)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D8 — default: no for MVP
  (FP32 only per D3).

## Decision
No kernel in this repo is templated on element type for MVP. Every
kernel signature takes `float*` directly (see ADR 0003). This is a
direct, mechanical consequence of D3 and is recorded separately only
because the spec lists it as its own open decision.

## Consequences
- Kernel code stays free of template boilerplate, which keeps it closer
  to what would be written on a whiteboard in an interview (hard
  constraint 6).
- If a post-MVP stretch effort adds a second dtype, the expected path is
  to templatize (`template <typename T>`) at that point rather than
  duplicate files by hand — but that refactor is explicitly out of scope
  now and not pre-built "just in case".
