# ADR 0004: FP32 First, FP16/BF16 Deferred (D4)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D4 -- default: "FP32 first, then FP16/BF16."

## Context
Attention is commonly run in FP16/BF16 in production for throughput, but
introducing reduced-precision dtypes multiplies tolerance rules and adds a
second, harder-to-reason-about source of numerical error on top of the
softmax-stability story this project is already about (spec SS1.4, SS1.7).
The spec's test-matrix section explicitly sequences this: "FP32 baseline;
optional FP16/BF16 after FP32 correctness."

## Decision
Phases 0-1 (and, by extension, every variant through at least V4) operate
on **`torch.float32` only**. `torch.float16`/`torch.bfloat16` inputs raise
`NotImplementedError` with a message pointing at this ADR
(`flashlite.reference.attention`'s `_check_shape_contract`); the CUDA
extension's `TORCH_CHECK` similarly rejects any non-`float32` dtype with a
specific message. `dtype` is nonetheless recorded as an explicit field on
every `BenchResult` (`"fp32"`, `flashlite/bench_schema.py`) and in the JSON
Schema (`benchmarks/schema/bench_result.schema.json`), matching
KernelForge's ADR 0003 precedent, so the schema does not need to change
shape when/if a later phase adds another dtype.

Correctness tolerances for FP32 are fixed at `atol=1e-5, rtol=1e-4`
(`flashlite.compare.DEFAULT_ATOL/DEFAULT_RTOL`), matching KernelForge's
`kDefaultAtol`/`kDefaultRtol` exactly for cross-project consistency.

## Consequences
- No templated/dtype-generic kernel code exists yet -- `attention_naive.cu`
  operates on `float*` directly, keeping the kernel bodies simple and
  interview-explainable (spec hard constraint 3: "keep the optimized
  implementation readable").
- The extreme-logit stability tests (`make_extreme_qkv`,
  `test_extreme_logits_do_not_overflow`,
  `test_naive_matches_reference_on_extreme_logits`) are meaningful FP32
  stress tests today; FP16/BF16 will need their own, tighter-magnitude
  extreme-logit cases in whatever later phase adds them, since FP16's
  exponent range overflows at a much smaller magnitude than FP32's.
- Anywhere this project's dtype support needs to expand, it is a two-file
  change in spirit (loosen the dtype check + add a dtype-appropriate
  tolerance), the same shape KernelForge's ADR 0003 anticipated for its own
  eventual dtype expansion.
