"""Unit tests for forgelm.evaluation.perplexity.evaluate_loss (FR9)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from forgelm.config import ModelConfig
from forgelm.dataset import NextTokenDataset
from forgelm.evaluation import evaluate_loss
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed


def _tiny_model_and_dataset() -> tuple[TransformerDecoder, ModelConfig, NextTokenDataset]:
    config = ModelConfig(
        vocab_size=24, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0
    )
    set_seed(0)
    model = TransformerDecoder(config)
    tokens = [i % config.vocab_size for i in range(200)]
    dataset = NextTokenDataset(tokens, config.context_length)
    return model, config, dataset


def test_evaluate_loss_matches_manual_cross_entropy_computation() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    model.eval()

    # Independent manual computation over the exact same (unshuffled)
    # windows evaluate_loss walks, as a cross-check of the function under
    # test rather than of itself.
    total_loss_sum = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(min(len(dataset), 8)):  # batch_size=8 below -> first batch only
            x, y = dataset[i]
            logits = model(x.unsqueeze(0))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss_sum += loss.item()
            total_tokens += y.numel()
    expected_loss = total_loss_sum / total_tokens

    result = evaluate_loss(model, dataset, batch_size=8, max_tokens=8 * dataset.context_length)
    assert result.loss == pytest.approx(expected_loss, rel=1e-4)
    assert result.perplexity == pytest.approx(math.exp(expected_loss), rel=1e-4)


def test_evaluate_loss_respects_token_budget() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    context_length = dataset.context_length
    tokens_per_batch = 4 * context_length

    small = evaluate_loss(model, dataset, batch_size=4, max_tokens=tokens_per_batch)
    large = evaluate_loss(model, dataset, batch_size=4, max_tokens=tokens_per_batch * 3)

    assert small.tokens_evaluated == tokens_per_batch
    assert small.num_batches == 1
    assert large.tokens_evaluated == tokens_per_batch * 3
    assert large.num_batches == 3


def test_evaluate_loss_never_overshoots_a_budget_that_is_not_batch_aligned() -> None:
    """max_tokens that doesn't land on an exact batch boundary must still
    be honored exactly -- evaluate_loss must truncate the crossing batch
    rather than counting all of it."""
    model, _config, dataset = _tiny_model_and_dataset()
    context_length = dataset.context_length  # 8
    # batch_size=4 -> 32 tokens/batch; ask for a budget that is not a
    # multiple of that (40 sits strictly between one batch and two).
    max_tokens = 40
    assert max_tokens % (4 * context_length) != 0

    result = evaluate_loss(model, dataset, batch_size=4, max_tokens=max_tokens)
    assert result.tokens_evaluated == max_tokens

    # Cross-check against an independent manual computation over exactly
    # the first 40 target tokens (unshuffled order), truncated mid-batch.
    total_loss_sum = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            if total_tokens >= max_tokens:
                break
            x, y = dataset[i]
            logits = model(x.unsqueeze(0))
            y_flat = y.reshape(-1)
            remaining = max_tokens - total_tokens
            if remaining < y_flat.numel():
                y_flat = y_flat[:remaining]
                logits = logits.reshape(-1, logits.size(-1))[:remaining]
            else:
                logits = logits.reshape(-1, logits.size(-1))
            loss = F.cross_entropy(logits, y_flat, reduction="sum")
            total_loss_sum += loss.item()
            total_tokens += y_flat.numel()
    expected_loss = total_loss_sum / total_tokens

    assert result.loss == pytest.approx(expected_loss, rel=1e-4)


def test_evaluate_loss_none_budget_covers_the_whole_dataset() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    result = evaluate_loss(model, dataset, batch_size=8, max_tokens=None)
    expected_batches = math.ceil(len(dataset) / 8)
    assert result.num_batches == expected_batches
    assert result.tokens_evaluated == len(dataset) * dataset.context_length


def test_evaluate_loss_restores_model_training_mode() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    model.train()
    evaluate_loss(model, dataset, batch_size=8)
    assert model.training is True

    model.eval()
    evaluate_loss(model, dataset, batch_size=8)
    assert model.training is False


def test_evaluate_loss_uses_no_grad() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    evaluate_loss(model, dataset, batch_size=8)
    for p in model.parameters():
        assert p.grad is None


def test_evaluate_loss_rejects_non_positive_batch_size() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_loss(model, dataset, batch_size=0)


def test_evaluate_loss_rejects_non_positive_max_tokens() -> None:
    model, _config, dataset = _tiny_model_and_dataset()
    with pytest.raises(ValueError, match="max_tokens"):
        evaluate_loss(model, dataset, batch_size=8, max_tokens=0)


def test_evaluate_loss_uniform_untrained_linear_head_is_near_log_vocab_size() -> None:
    """A model that always outputs uniform logits should have loss ==
    log(vocab_size) exactly -- a sanity anchor independent of any trained
    model's specific numbers."""

    class _UniformStub(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self._unused = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len = x.shape
            return torch.zeros(batch, seq_len, self.vocab_size)  # equal logits -> uniform softmax

    vocab_size = 24
    tokens = [i % vocab_size for i in range(100)]
    dataset = NextTokenDataset(tokens, context_length=8)
    result = evaluate_loss(_UniformStub(vocab_size), dataset, batch_size=8)
    assert result.loss == pytest.approx(math.log(vocab_size), abs=1e-5)
