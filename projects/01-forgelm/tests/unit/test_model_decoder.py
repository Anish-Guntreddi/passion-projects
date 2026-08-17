from __future__ import annotations

import pytest
import torch

from forgelm.config import ConfigError, ModelConfig
from forgelm.model import (
    TransformerBlock,
    TransformerDecoder,
    count_parameters,
    expected_parameter_count,
    precompute_rope_angles,
)


def _tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = dict(
        vocab_size=64,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


# -- ModelConfig validation ---------------------------------------------------------


def test_model_config_rejects_indivisible_d_model_n_heads() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        _tiny_config(d_model=30, n_heads=4)


def test_model_config_rejects_odd_head_dim() -> None:
    # d_model=12 / n_heads=4 -> head_dim=3 (odd) -- RoPE needs an even head_dim.
    with pytest.raises(ConfigError, match="even"):
        _tiny_config(d_model=12, n_heads=4)


def test_model_config_accepts_n_kv_heads_equal_to_n_heads() -> None:
    cfg = _tiny_config(n_heads=4, n_kv_heads=4)
    assert cfg.n_kv_heads == 4


def test_model_config_rejects_gqa_not_equal_n_heads() -> None:
    with pytest.raises(NotImplementedError, match="grouped-query"):
        _tiny_config(n_heads=4, n_kv_heads=2)


def test_model_config_rejects_bad_dtype() -> None:
    with pytest.raises(ConfigError, match="dtype"):
        _tiny_config(dtype="int8")


# -- shapes -------------------------------------------------------------------------


def test_decoder_output_shape() -> None:
    config = _tiny_config()
    model = TransformerDecoder(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(token_ids)
    assert logits.shape == (2, 10, config.vocab_size)


def test_decoder_rejects_sequence_longer_than_context_length() -> None:
    config = _tiny_config(context_length=8)
    model = TransformerDecoder(config)
    token_ids = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError, match="context_length"):
        model(token_ids)


def test_decoder_dropout_zero_is_deterministic_in_eval_mode() -> None:
    config = _tiny_config(dropout=0.0)
    model = TransformerDecoder(config)
    model.eval()
    token_ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        out_a = model(token_ids)
        out_b = model(token_ids)
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


def test_transformer_block_output_shape() -> None:
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    cos, sin = precompute_rope_angles(head_dim=4, max_seq_len=5)
    x = torch.randn(2, 5, 16)
    out = block(x, cos, sin)
    assert out.shape == x.shape


# -- parameter-count sanity (spec §1.8) ----------------------------------------------


@pytest.mark.parametrize("tie_weights", [True, False])
def test_decoder_parameter_count_matches_analytic_formula(tie_weights: bool) -> None:
    config = _tiny_config(tie_weights=tie_weights)
    model = TransformerDecoder(config)
    assert count_parameters(model) == expected_parameter_count(config)


def test_untied_model_has_more_parameters_than_tied() -> None:
    tied = TransformerDecoder(_tiny_config(tie_weights=True))
    untied = TransformerDecoder(_tiny_config(tie_weights=False))
    assert (
        count_parameters(untied)
        == count_parameters(tied) + tied.config.vocab_size * tied.config.d_model
    )


def test_weight_tying_shares_tensor_and_parameter_count() -> None:
    config = _tiny_config(tie_weights=True)
    model = TransformerDecoder(config)
    assert model.output_proj.weight is model.token_embedding.weight
    # nn.Module.parameters() deduplicates by object identity, so the tied
    # weight is counted exactly once (see docs/decisions/0006-weight-tying.md).
    param_ids = {id(p) for p in model.parameters()}
    named = dict(model.named_parameters())
    assert "output_proj.weight" not in named  # not a *separate* named parameter
    assert len(param_ids) == len(named)


def test_larger_model_has_more_parameters() -> None:
    small = TransformerDecoder(_tiny_config(d_model=16, d_ff=32, n_layers=1))
    big = TransformerDecoder(_tiny_config(d_model=32, d_ff=64, n_layers=2))
    assert count_parameters(big) > count_parameters(small)
