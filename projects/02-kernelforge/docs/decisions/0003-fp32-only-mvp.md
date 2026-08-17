# ADR 0003: FP32-Only MVP (D3)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D3 — default: yes.

## Context
Supporting multiple dtypes (fp16/bf16/tf32/fp64) multiplies the number of
kernel variants, reference implementations, and tolerance rules without
changing the memory-hierarchy lessons Phases 0–1 are built to teach.

## Decision
All kernels, references, tests, and benchmarks in this repo operate on
**`float` (FP32) only** for MVP. `dtype` is nonetheless recorded as an
explicit field in the benchmark schema (`BenchResult::dtype`, currently
always `"fp32"`) so the schema does not need to change shape when/if a
later, explicitly-scoped stretch effort adds another dtype.

## Consequences
- Reference implementations (`kernelforge::reference::*`) and tolerance
  defaults (`kernelforge::allclose`, ADR 0002-adjacent) are written once,
  for FP32, with `atol=1e-5f`, `rtol=1e-4f` defaults appropriate to
  single-precision accumulation error on sums up to the sizes used here.
- No templated kernel code exists yet (see ADR 0008); this keeps kernel
  bodies simple and interview-explainable (hard constraint 6).
