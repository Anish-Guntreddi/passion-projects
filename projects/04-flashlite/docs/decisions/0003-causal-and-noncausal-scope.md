# ADR 0003: Support Both Causal and Non-Causal Attention From Phase 0 (D3)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D3 -- default: "both, causal via flag; if scope pressure, causal-only with ADR."

## Context
Causal (autoregressive, decoder-style) attention is the more common
production use case and the one the resume narrative (spec SS1.6)
emphasizes, but non-causal (encoder-style, full bidirectional) attention is
the simpler case to get right first and is a strict subset of the same
masking logic (non-causal is "no mask applied"). The spec allows narrowing
to causal-only under scope pressure, with an ADR recording that narrowing.

## Decision
**No narrowing was needed.** Both modes are supported from Phase 0 onward,
selected by a single `causal: bool` parameter threaded identically through
every variant (`reference_attention(..., causal=...)`,
`attention_naive_forward(q, k, v, causal)`). This was not a scope-pressure
casualty: causal is implemented as one additional masking step
(`docs/attention-math.md` SS3) on top of the same score/softmax/weighted-sum
computation non-causal already needs, so supporting both costs one `if`
per kernel (`compute_scores_kernel`'s `if (causal && j > i)`) and one
mask-construction step in the Python reference (`torch.triu`), not a
second implementation.

## Consequences
- Every correctness test in this project is parametrized over
  `causal in (False, True)` (see
  `tests/math/test_reference_attention.py`,
  `tests/correctness/test_naive_attention.py`), not written once and
  assumed to generalize.
- The causal-specific behavioral invariant ("row `i`'s output cannot depend
  on key/value positions `j > i`") is tested directly by perturbing future
  positions and checking the earlier output is unchanged
  (`test_causal_output_does_not_depend_on_future_keys`), not just by
  inspecting that the mask *looks* triangular.
- If a future phase's tiling/online-softmax work turns out to make one mode
  substantially harder to get correct than the other, that phase can note
  the asymmetry in its own decision record -- this ADR only commits Phases
  0-1 to both modes, which is already delivered.
