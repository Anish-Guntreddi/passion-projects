# ADR 0006: torch SDPA as the Platform-Optimized Comparison Backend (D7)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D7 -- default: "torch.nn.functional.scaled_dot_product_attention."

## Context
The spec's benchmark plan (SS1.8) requires a "platform optimized backend
(SDPA)" in the comparison set alongside the PyTorch reference, naive,
tiled, and online/fused variants -- primarily a **Phase 7** concern
("Framework comparison"). Which backend to use for that role was an open
decision (D7) with a stated default.

## Decision
`torch.nn.functional.scaled_dot_product_attention` is the designated
platform-optimized comparison backend for the Phase 7 benchmark ladder.

Ahead of Phase 7, it is put to a second, narrower use already in Phase 0:
`tests/math/test_reference_attention.py::test_cross_check_against_torch_sdpa`
compares `flashlite.reference.attention` (V0) against SDPA (both causal and
non-causal) as an **independent second opinion** on V0's correctness --
NOT as a replacement for V0's status as ground truth. V0 remains the
project's actual reference implementation (checked independently against a
from-scratch, no-`torch.matmul` Python-loop brute-force computation in the
same file), per the spec's hard constraint 2 ("derive algorithms from
documented math and validate incrementally") -- SDPA is a highly-optimized,
opaque library kernel, exactly the kind of implementation this project's
non-goals say not to reproduce or depend on as its source of truth
("reproducing the production FlashAttention library" is explicitly out of
scope; SDPA itself may be backed by exactly such a fused kernel depending
on the installed torch/hardware).

## Consequences
- Phase 0 gets one extra, free correctness signal on V0 (agreement between
  two independently-implemented softmax-attention computations, one
  hand-rolled and one from a heavily-tested library) at negligible cost.
- Phase 7's benchmark ladder has its comparison backend decided before that
  phase starts, so no benchmark-methodology decision is made under time
  pressure while numbers are already being collected.
- If a future torch version changes SDPA's numerics (e.g. a different
  fused-kernel backend becomes the default on some hardware), the Phase 0
  cross-check test would surface that as a test failure -- an intentional
  tripwire, not a maintenance burden to route around by loosening the
  tolerance.
