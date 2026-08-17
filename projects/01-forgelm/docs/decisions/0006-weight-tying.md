# ADR 0006: Weight tying in the MVP (D4)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D4 (`01-forgelm-spec.md` §1.9) — *"Weight tying in MVP →
default: yes (simple, standard)."*

## Decision

`forgelm.model.TransformerDecoder` ties the output projection's weight
matrix to the token embedding matrix by default (`ModelConfig.tie_weights
= True`):

```python
if config.tie_weights:
    self.output_proj.weight = self.token_embedding.weight
```

Both attributes reference the same `nn.Parameter` object. `tie_weights =
False` remains available (constructs a second, independently-learned
`vocab_size x d_model` matrix) for anyone who wants to A/B it, but is not
the default.

## Rationale

- Press & Wolf (2017), "Using the Output Embedding to Improve Language
  Models": tying the input and output embeddings is standard practice in
  small-to-medium decoder-only LMs and reduces total parameter count
  meaningfully at ForgeLM's model scale, where `vocab_size x d_model` is
  often the single largest tensor in the model (e.g. at `vocab_size=512,
  d_model=64`, tying removes ~32K of a model with well under 1M total
  parameters).
- It is the simpler default per the spec's own guidance ("simple,
  standard") and does not need any custom machinery: `nn.Module`
  naturally supports two attribute names pointing at one `nn.Parameter`,
  and both `state_dict()` and `parameters()` deduplicate it automatically
  (verified by `tests/unit/test_model_decoder.py::
  test_weight_tying_shares_tensor_and_parameter_count`).

## Alternatives considered

- **Untied embeddings** (`tie_weights=False`): strictly more expressive
  (separate input/output representations), at the cost of doubling the
  largest parameter tensor. Kept as an explicit opt-out, not removed,
  since Phase 5's scaling experiments may want to compare the two, but it
  is not the MVP default.

## Consequences

- `forgelm.model.decoder.count_parameters()` (a thin wrapper over
  `sum(p.numel() for p in model.parameters())`) automatically reports the
  correct deduplicated count under tying, because `nn.Module.parameters()`
  deduplicates by Python object identity, not by attribute name.
- `forgelm.model.decoder.expected_parameter_count()` (the analytic
  cross-check used in tests) branches on `config.tie_weights` to add or
  omit the second `vocab_size x d_model` term.
- `nn.Module.state_dict()`, unlike `parameters()`, does **not**
  deduplicate: a tied model's `state_dict()` has *two* keys
  (`token_embedding.weight` and `output_proj.weight`), both viewing the
  same underlying storage (`state_dict()['token_embedding.weight'].
  data_ptr() == state_dict()['output_proj.weight'].data_ptr()` is `True`,
  verified interactively against this project's pinned torch build) but
  not the same Python tensor object. This is harmless for checkpointing:
  `load_state_dict` copies each key's values into the corresponding
  attribute in place, and since both attributes still reference the same
  `nn.Parameter` object (the tie was established at model construction,
  before `load_state_dict` runs), copying the same values in under both
  keys is redundant but not incorrect. `count_parameters()` /
  `expected_parameter_count()` deliberately use `.parameters()` (which
  *does* deduplicate), not `state_dict()`, for exactly this reason — see
  `tests/unit/test_model_decoder.py::
  test_weight_tying_shares_tensor_and_parameter_count`.
