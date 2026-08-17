# Model math

Status: covers Phase 4's generation/evaluation math (this page). The
full from-scratch derivation of the Phase 2 architecture (RMSNorm, RoPE,
causal self-attention, SwiGLU) referenced in the roadmap is a known
documentation gap left for Phase 6 (portfolio hardening) — the
implementations themselves exist and are documented inline (see the
module docstrings in `src/forgelm/model/`: `rmsnorm.py`, `rope.py`,
`attention.py`, `mlp.py`, `block.py`, `decoder.py`, each carrying the
derivation/citation for its component), just not yet consolidated into
this standalone page.

## Generation: decoding distributions (Phase 4, FR8)

Given the model's logits `z ∈ R^V` at the final sequence position
(`V` = vocab size), `forgelm.generation.sampling` implements four
decoding transformations, applied in this order:

1. **Temperature scaling**: `z' = z / T`. `T < 1` sharpens the
   distribution toward the argmax (more deterministic); `T > 1` flattens
   it (more random). `T = 1` is a no-op.
2. **Top-k filtering** (optional): keep only the `k` largest entries of
   `z'`, set the rest to `-inf`. Restricts sampling to a fixed-size
   candidate set regardless of how the probability mass is distributed.
3. **Top-p / nucleus filtering** (optional; Holtzman et al. 2019): sort
   `softmax(z')` descending, keep the smallest prefix whose cumulative
   probability is `>= p`, set the rest to `-inf`. Unlike top-k, the
   candidate set size adapts to how peaked or flat the distribution is at
   each step.
4. **Categorical sampling**: `token ~ Categorical(softmax(z'))`, drawn
   from a seeded `torch.Generator` so the same `(prompt, seed,
   temperature, top_k, top_p)` always reproduces the same continuation.

Greedy decoding (`do_sample=False`) skips all four steps and takes
`argmax(z)` directly — deterministic regardless of `seed`.

`generate()` re-runs the full forward pass over the current sequence
(cropped to the last `context_length` tokens) at every step — no KV
cache. This is a documented performance trade-off (see
`forgelm.generation.sampling`'s module docstring), not a correctness
concern: it produces the same output a KV-cached implementation would,
just with `O(n²)` instead of `O(n)` total forward-pass compute over an
`n`-token generation.

## Evaluation: loss and perplexity (Phase 4, FR9)

`forgelm.evaluation.perplexity.evaluate_loss` computes the token-weighted
mean cross-entropy loss over `N` target tokens:

```
loss = (1/N) * sum_i [ -log P(y_i | x_i) ]
     = (1/N) * sum_i [ cross_entropy(logits_i, y_i) ]     (reduction="sum", then divided by N)

perplexity = exp(loss)
```

Perplexity is the standard "effective branching factor" reading of
cross-entropy loss: a perplexity of `k` means the model is, on average,
as uncertain about the next token as if it were choosing uniformly among
`k` options. An untrained model with uniform logits over a vocabulary of
size `V` has `loss = log(V)` and `perplexity = V` exactly (verified in
`tests/unit/test_evaluation_perplexity.py::
test_evaluate_loss_uniform_untrained_linear_head_is_near_log_vocab_size`)
— the natural sanity anchor for whether a reported perplexity number is
even in a plausible range.

The mean is **token-weighted**, not batch-weighted: `evaluate_loss` sums
per-token loss and total token count separately across all batches (a
final, possibly-partial batch does not get equal weight to a full one).
This differs slightly from `Trainer.evaluate()`'s periodic
during-training validation signal (a mean of equal-sized batch means,
adequate for a quick training-time check); see
`forgelm.evaluation.perplexity`'s module docstring for the full
rationale.
