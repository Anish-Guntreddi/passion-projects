# ADR 0002: [Batch, Heads, Seq, HeadDim] Contiguous Layout Only (D2)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D2 -- default: "[B, H, S, D] contiguous only."

## Context
Attention kernels can support arbitrary strides (non-contiguous views from
upstream reshape/transpose/slicing) and multiple layout conventions
(`[B, H, S, D]` vs. `[B, S, H, D]` vs. packed QKV). Supporting either
generality multiplies indexing complexity in every kernel without changing
the memory-movement lessons this project exists to teach (spec SS1.1), and
the spec's own non-goals explicitly exclude "supporting every dtype/layout/
mask."

## Decision
Every variant (V0 through the eventual V4) operates on `q, k, v` tensors
shaped **exactly** `[batch, heads, seq_len, head_dim]`, and requires them
**contiguous** (`.is_contiguous()` in `bindings.cpp` for the CUDA
extension; the pure-Python V0 reference works correctly on non-contiguous
input too, but is not benchmarked or contract-tested in that configuration
since the CUDA path -- the one actually exercising this project's
memory-movement story -- requires it). Additionally, for the self-attention
MVP `q`, `k`, `v` must share one shape (`Sq == Sk == Sv`); cross-attention
with an independent query length is out of scope for Phases 0-1.

Enforcement:
- `flashlite.reference.attention._check_shape_contract` (Python `ValueError`
  with a specific message).
- `bindings.cpp`'s `TORCH_CHECK` sequence (CUDA `RuntimeError` with a
  specific message) -- see `tests/edge_cases/test_shape_contract.py` for
  every enforced case.

## Consequences
- Kernel index arithmetic (`attention_naive.cu`) can assume simple
  row-major strides (`((b*H+h)*S+i)*D+d`) throughout, with no stride
  parameters to thread through every kernel.
- Awkward, non-tile-multiple `seq_len` values remain fully supported (V1 is
  untiled; this ADR is about *layout*, not about size), verified directly
  by `tests/correctness/test_naive_attention.py`'s `seq_len` sweep
  (1, 2, 7, 33, 257).
- A caller with a non-contiguous or differently-laid-out tensor must call
  `.contiguous()` (and reshape/transpose to `[B, H, S, D]`) themselves
  before calling into this project's kernels; this project does not do it
  for them, so the caller is never silently paying for a hidden copy or
  silently getting a wrong answer from a layout it didn't expect.
