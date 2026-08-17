# ADR 0007: Grouped-query attention (GQA) deferred to stretch (D5)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D5 (`01-forgelm-spec.md` §1.9) — *"GQA in MVP or
stretch → default: stretch."* Also FR4: *"GQA may be Phase 2 / stretch."*

## Decision

Phase 2's `CausalSelfAttention` implements standard multi-head attention
only: one key/value head per query head. Grouped-query attention (fewer
key/value heads shared across groups of query heads, Ainslie et al. 2023)
is **not implemented** in this MVP.

`ModelConfig.n_kv_heads` exists as a field (FR5 explicitly lists it in
the config surface) but is constrained by `ModelConfig.__post_init__` to
either `None` or exactly `n_heads`; any other value raises
`NotImplementedError` with a message pointing at this ADR, rather than
silently being ignored or (worse) silently doing something other than
what was asked.

## Rationale

- The spec's own default is "stretch" — GQA is explicitly optional scope,
  and portfolio-wide rule 3 ("correctness before optimization") argues for
  landing standard multi-head attention correctly and thoroughly-tested
  first.
- GQA's main benefit (smaller KV-cache during autoregressive generation)
  is not yet relevant: generation/KV-caching is Phase 4, not Phase 2/3.
  Implementing GQA now, before there is a caching consumer that benefits
  from it, would be optimizing ahead of the feature that motivates it.
- Reserving the config field now (rather than adding it later) means
  Phase 4/5 can implement GQA as a config-compatible extension without a
  breaking `ModelConfig` schema change, while every Phase 2/3 checkpoint
  and config stays forward-compatible.

## Consequences

- `tests/unit/test_model_decoder.py::
  test_model_config_rejects_gqa_not_equal_n_heads` locks in the
  "reject, don't silently ignore" behavior.
- If a future phase implements GQA, this ADR should be superseded (or
  amended) with the chosen key/value-head-grouping scheme, and the
  `NotImplementedError` in `ModelConfig.__post_init__` relaxed
  accordingly.
