"""Full decoder-only Transformer.

    token ids (B, T)
        -> token embedding                       (B, T, d_model)
        -> N x TransformerBlock (RoPE inside attention)
        -> final RMSNorm
        -> output projection (tied to the embedding by default, D4)
        -> logits (B, T, vocab_size)

This is the only place in ``forgelm.model`` that owns the full forward
pass end to end; everything else (RMSNorm, RoPE, attention, SwiGLU,
TransformerBlock) is architecture-only building blocks it composes.
"""

from __future__ import annotations

import torch
from torch import nn

from forgelm.config import ModelConfig
from forgelm.model.block import TransformerBlock
from forgelm.model.rmsnorm import RMSNorm
from forgelm.model.rope import precompute_rope_angles

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPE_BY_NAME)}")
    return _DTYPE_BY_NAME[name]


class TransformerDecoder(nn.Module):
    """A GPT/LLaMA-style decoder-only Transformer, built entirely from the
    architecture modules in this package (no ``transformers``-library
    model class involved anywhere)."""

    # Declared here (rather than left to register_buffer()'s inference)
    # so static type checkers see these as Tensor attributes -- nn.Module
    # itself types arbitrary buffer names as `Module | Tensor`, which
    # makes `self.rope_cos[:seq_len]` in forward() an indexing error under
    # pyright even though it is always a Tensor at runtime.
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.d_model // config.n_heads

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    dropout=config.dropout,
                    rmsnorm_eps=config.rmsnorm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rmsnorm_eps)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # Weight tying (D4, docs/decisions/0006-weight-tying.md): the
            # output projection reuses the token embedding matrix instead
            # of learning a separate one (Press & Wolf, 2017). Assigning
            # the same nn.Parameter object to both attributes is what
            # nn.Module.parameters()/state_dict() naturally deduplicate --
            # see count_parameters() below.
            self.output_proj.weight = self.token_embedding.weight

        cos, sin = precompute_rope_angles(head_dim, config.context_length, theta=config.rope_theta)
        # persistent=False: these are a deterministic, cheap-to-recompute
        # function of (head_dim, context_length, rope_theta) -- already
        # captured in model_config -- so they don't need to bloat
        # checkpoints or state_dicts.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        dtype = resolve_dtype(config.dtype)
        if dtype is not torch.float32:
            self.to(dtype)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # A small fixed std (GPT-2-style init), applied uniformly to every
        # Linear/Embedding weight. Not tuned for scale -- appropriate for
        # this project's small toy/portfolio-scale models (Phase 5 scaling
        # experiments may revisit this if it becomes a bottleneck).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids shape (batch, seq_len), got {token_ids.shape}")
        _batch, seq_len = token_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length {self.config.context_length}"
            )
        x = self.token_embedding(token_ids)
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Total *unique* parameter count -- weight-tied tensors counted once.

    ``nn.Module.parameters()`` already deduplicates by object identity
    (the same ``nn.Parameter`` object reachable under two attribute names,
    as weight tying does, is yielded only once), so this is just a sum
    over it -- verified against :func:`expected_parameter_count` in
    ``tests/unit/test_model_decoder.py``.
    """
    return sum(p.numel() for p in model.parameters())


def expected_parameter_count(config: ModelConfig) -> int:
    """Analytic parameter-count formula, used as an independent
    cross-check of :func:`count_parameters` in tests (spec §1.8
    "parameter-count sanity").

    Per block (no biases anywhere in this architecture):
      - attention: qkv_proj (d_model x 3*d_model) + out_proj (d_model x d_model)
                   = 4 * d_model^2
      - SwiGLU MLP: gate_proj + up_proj (d_model x d_ff each) + down_proj (d_ff x d_model)
                   = 3 * d_model * d_ff
      - two RMSNorm gains (attn_norm, mlp_norm): 2 * d_model

    Plus: token embedding (vocab_size x d_model), the final RMSNorm gain
    (d_model), and the output projection -- 0 extra parameters if
    weight-tied (D4), otherwise another vocab_size x d_model.
    """
    d_model, d_ff, n_layers = config.d_model, config.d_ff, config.n_layers
    per_block = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model
    total = config.vocab_size * d_model + n_layers * per_block + d_model  # + final norm gain
    if not config.tie_weights:
        total += config.vocab_size * d_model
    return total
