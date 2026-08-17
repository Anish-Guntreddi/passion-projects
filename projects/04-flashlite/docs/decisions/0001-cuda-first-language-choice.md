# ADR 0001: CUDA C++ Primary, Triton Deferred to Later Comparison (D1)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D1 -- default: "CUDA C++ primary; Triton as later comparison phase."

## Context
The spec's variant ladder (SS1.3) is language-agnostic through V4 (naive ->
tiled -> online-softmax -> fused) and adds an explicitly optional V5
("Triton (or CUDA) counterpart to compare programming models"). Writing
V1-V4 in CUDA C++ vs. Triton is an open decision (D1) with a stated
recommended default.

## Decision
Every kernel through V4 (naive, tiled, online-softmax, fused) is written in
CUDA C++, compiled via `torch.utils.cpp_extension` (ADR 0005). Triton is
explicitly deferred to Phase 7 ("Framework comparison") as an *optional*
V5 counterpart, written only after V1-V4 already exist and only to compare
the two programming models directly on the same algorithm -- not as a
substitute for any CUDA variant.

## Consequences
- Every algorithmic step (tiling, online-softmax accumulation, fusion) is
  first understood and derived at the CUDA level, matching the spec's hard
  constraint 2 ("derive algorithms from documented math and validate
  incrementally") and its non-goal of not starting with a higher-level
  abstraction before the portable baseline works.
- Target architecture is pinned to **sm_89 (RTX 4090)**, matching
  KernelForge's ADR 0001
  (`../02-kernelforge/docs/decisions/0001-target-gpu-architecture.md`) --
  the only GPU this repo is built and tested against. `setup.py` sets
  `TORCH_CUDA_ARCH_LIST=8.9` accordingly (overridable via the standard env
  var).
- If Phase 7's Triton comparison is skipped under scope pressure, nothing
  in Phases 0-6 needs to change -- V5 is additive, not load-bearing for any
  earlier exit criterion.
