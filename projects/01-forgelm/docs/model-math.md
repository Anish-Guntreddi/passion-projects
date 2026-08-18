# Model math

Status: covers the full Phase 2 architecture (RMSNorm, RoPE, causal
self-attention, SwiGLU, and how they compose into one decoder block and
the full decoder) and Phase 4's generation/evaluation math. Every formula
below is transcribed from the actual implementation in `src/forgelm/model/`
(cross-checked against the module docstrings, not derived independently
of the code) — variable names match the source so a reviewer can read
this page next to the code with no translation step. Shapes use `B` =
batch size, `T` = sequence length, `C` = `d_model`, `H` = `n_heads`,
`Hd` = `head_dim = C / H`, `V` = `vocab_size`.

## RMSNorm (`forgelm.model.rmsnorm.RMSNorm`)

Zhang & Sennrich (2019), "Root Mean Square Layer Normalization". Unlike
LayerNorm, RMSNorm does not re-center the input (no mean subtraction) —
it only rescales by the root-mean-square of the activations, then applies
a learned per-channel gain `weight ∈ R^C` (initialized to `1`, so training
starts as an identity rescale):

```
rms(x)     = sqrt( mean_i(x_i^2) + eps )        (mean over the last dim, size C)
RMSNorm(x) = (x / rms(x)) * weight
```

`eps` (`rmsnorm_eps` in `ModelConfig`, default `1e-6`) guards the
`sqrt`/division against a near-zero activation vector. The reduction runs
in float32 regardless of the input's dtype (cast up, normalize, cast back
down) — `mean(x_i^2)` loses precision fast in bf16/fp16, and the extra
cast is cheap next to the matmuls surrounding it in a `TransformerBlock`.
This is what every modern open decoder-only LLM (LLaMA, Mistral, Gemma,
...) uses in place of LayerNorm. Verified against a direct re-derivation
from the formula (not the module under test) in
`tests/unit/test_model_rmsnorm.py`.

## RoPE — rotary position embeddings (`forgelm.model.rope`)

Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position
Embedding". Instead of adding a position vector to token embeddings, RoPE
rotates each query/key vector in `Hd / 2` independent 2D sub-planes by an
angle proportional to sequence position, so the dot product between a
rotated query at position `i` and a rotated key at position `j` depends
only on the relative offset `i - j` and the two vectors' original content
— relative position is encoded directly into the attention score, with no
additive position embedding anywhere. This is the positional scheme used
by LLaMA, Mistral, GPT-NeoX, and most modern open decoder-only LLMs.

`forgelm.model.rope` implements the **rotate-half** formulation (splits
`Hd` into two contiguous halves, rather than interleaving adjacent
coordinate pairs as the original paper does — mathematically equivalent,
just a different choice of which coordinates get paired; this is the
formulation LLaMA and most modern implementations use):

```
freq_k       = theta ** (-2k / Hd),   k = 0 .. Hd/2 - 1        (theta = rope_theta, default 10000.0)
angle(p, k)  = p * freq_k                                       (p = sequence position, 0-indexed)

rotate_half([x1, x2]) = [-x2, x1]        (x1, x2: first/second half of x's last dim, each size Hd/2)
RoPE(x, p) = x * cos(angle(p)) + rotate_half(x) * sin(angle(p))
```

`precompute_rope_angles(head_dim, max_seq_len, theta)` builds `cos`/`sin`
tables once, shape `(max_seq_len, Hd)` (each half-size table duplicated
across both halves so it broadcasts directly against a full `Hd`-length
vector), cached as non-persistent buffers on `TransformerDecoder`
(`rope_cos`/`rope_sin` — deterministic function of `(head_dim,
context_length, rope_theta)`, all already in `ModelConfig`, so they don't
need to be saved in checkpoints). `apply_rope` is called once for `q` and
once for `k` inside attention, per-head (RoPE never mixes information
*across* heads — each head's `Hd`-vector is rotated independently); `v`
is left untouched. Shape/determinism verified in
`tests/unit/test_model_rope.py`.

## Causal multi-head self-attention (`forgelm.model.attention.CausalSelfAttention`)

Implemented directly with tensor reshapes and matmuls — not
`torch.nn.MultiheadAttention` or
`torch.nn.functional.scaled_dot_product_attention` — so the multi-head
split/merge and causal masking (FR4) stay this project's own, readable
code (spec §1.7: "no copied full model implementation").

```
x                (B, T, C)
qkv = qkv_proj(x)                        (B, T, 3C)  -- one fused Linear, no bias
q, k, v = qkv.split(C, dim=-1)           each (B, T, C)

split_heads(t) = t.view(B, T, H, Hd).transpose(1, 2)     -> (B, H, T, Hd)
q, k, v = split_heads(q), split_heads(k), split_heads(v)

q = apply_rope(q, cos, sin)              (B, H, T, Hd) -- rotated per-head
k = apply_rope(k, cos, sin)

scores  = (q @ k^T) / sqrt(Hd)                              (B, H, T, T)
scores  = scores.masked_fill(causal_mask, -inf)             causal_mask = strictly-upper triangle
weights = softmax(scores, dim=-1)
out     = weights @ v                                       (B, H, T, Hd)

out = out.transpose(1, 2).reshape(B, T, C)                  merge heads back
out = out_proj(out)                                          (B, T, C), no bias
```

The `1/sqrt(Hd)` scale (Vaswani et al. 2017) keeps the pre-softmax score
variance roughly constant as `Hd` grows, so softmax doesn't saturate into
a near-one-hot distribution purely as an artifact of head width.
`causal_mask = torch.triu(ones(T, T), diagonal=1)` is `True` at position
`(i, j)` exactly when `j > i` (a strictly-future key relative to query
position `i`) — filling those positions with `-inf` before the softmax
sends their post-softmax weight to (numerically) `0`, which is how the
decoder-only "cannot see the future" property is enforced (verified in
`tests/unit/test_model_attention.py` by checking that changing a *future*
token's embedding does not change the current position's output). This
project's attention has **no KV cache** anywhere — `generate()` re-runs
the full forward pass every step (see below); the block above is the
entire attention computation for a full `(B, T, C)` input, training or
inference alike.

## SwiGLU gated MLP (`forgelm.model.mlp.SwiGLUMLP`)

Shazeer (2020), "GLU Variants Improve Transformer". In place of a
standard two-layer MLP (`Linear -> activation -> Linear`), SwiGLU uses
three projections and a gated (elementwise-multiplied) hidden state:

```
SiLU(z)     = z * sigmoid(z)                    (also called "Swish")
SwiGLU(x)   = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

`gate_proj`, `up_proj`: `C -> d_ff`; `down_proj`: `d_ff -> C`; none carry
a bias. This is the MLP used by LLaMA, Mistral, and most modern open
decoder-only LLMs in place of GELU/ReLU MLPs (FR4: "SwiGLU (or equivalent
gated MLP)"). `F.silu` is a plain elementwise activation function — the
same tier of PyTorch primitive as `torch.softmax` or `nn.Linear` used
elsewhere in this package, not a "copied model implementation."

## One decoder block (`forgelm.model.block.TransformerBlock`)

Pre-norm residual composition of the two sub-layers above:

```
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Pre-norm (normalize *before* each sub-layer, rather than after it and the
residual add) is used because it is what every modern decoder-only LLM
(GPT-2 onward, LLaMA, ...) uses — it keeps gradients well-behaved through
deep stacks without needing the delicately-tuned learning-rate warmup the
original post-norm Transformer required. Each block owns two independent
`RMSNorm` instances (`attn_norm`, `mlp_norm`) with their own learned
gains — they are not shared.

## Full decoder (`forgelm.model.decoder.TransformerDecoder`)

```
token_ids (B, T)
    -> token_embedding                         (B, T, C)     nn.Embedding(V, C)
    -> [TransformerBlock] x n_layers            (B, T, C)     RoPE angles shared across all blocks
    -> final_norm (RMSNorm)                     (B, T, C)
    -> output_proj                              (B, T, V)     nn.Linear(C, V, bias=False)
```

`output_proj.weight` is the **same `nn.Parameter` object** as
`token_embedding.weight` when `tie_weights=True` (D4, the MVP default —
see `docs/decisions/0006-weight-tying.md`; Press & Wolf 2017), not a
separate tensor initialized to equal values — `nn.Module.parameters()`
deduplicates by object identity, so a tied model has exactly one
`V x C` embedding/output matrix, not two. `expected_parameter_count(config)`
is an independent closed-form parameter count (not reading
`count_parameters`'s own code path) used to cross-check every model
actually built in this project, including the three Phase 5 scaling
configs:

```
per_block  = 4*C^2 + 3*C*d_ff + 2*C     (attention: 4C^2; SwiGLU: 3*C*d_ff; two RMSNorm gains: 2C)
total      = V*C + n_layers*per_block + C            (+ final RMSNorm gain: C)
             + (V*C if not tie_weights else 0)        (untied output head, if configured)
```

Weight initialization (`TransformerDecoder._init_weights`): every
`nn.Linear`/`nn.Embedding` weight is drawn from `Normal(0, 0.02)`
(GPT-2-style fixed-std init) — no `Linear` in this architecture carries a
bias (`qkv_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`,
`output_proj` are all `bias=False`), so there is nothing to zero-init.
This is not scale-tuned for depth/width — appropriate for this project's
toy/portfolio-scale models (164K–6.4M parameters in the Phase 5 scaling
experiment), not claimed suitable for a much deeper stack without
revisiting it.

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
