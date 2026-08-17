# ADR 0002: Minimum Supported Compute Capability (D2)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D2 — default: whatever D1's card
  supports; guard newer features behind capability checks.

## Context
The spec's recommended default ties the minimum supported compute
capability to whatever the developer's card provides (ADR 0001: sm_89),
rather than inventing portability to hardware that cannot be tested.

## Decision
Minimum supported compute capability is **8.9** (the RTX 4090 itself).
`kernelforge::require_capability_or_throw(info, 8, 9)` is called at the
top of every executable entry point (`device-info`, every `bench_*`
binary) before any kernel launch. If the queried device reports a lower
capability, the program prints a clear error naming the required and
actual capability and exits non-zero — it does not attempt to run a
"maybe it still works" path (hard constraint 8: fail loudly on
unsupported hardware).

Nothing in this repo currently *uses* a feature newer than 8.9 (e.g.
Hopper-only intrinsics), so 8.9 is simultaneously the floor and the
ceiling for MVP. If a later phase adds an sm_90+-only feature, that
feature must be compiled behind `#if __CUDA_ARCH__ >= 900` and gated by
a runtime capability check, not assumed.

## Consequences
- No multi-architecture fat binary is built for MVP (`CMAKE_CUDA_ARCHITECTURES=89`
  only) — this is a laboratory pinned to one card, not a portable library.
- Capability checks are cheap (one `cudaGetDeviceProperties` call) and run
  once per process, so they add no measurable overhead to benchmarks.
