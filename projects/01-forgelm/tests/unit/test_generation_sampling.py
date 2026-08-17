"""Unit tests for forgelm.generation.sampling: decoding-mode constraints
(spec §1.8 "sampling constraints") on controlled logits, independent of
any real model."""

from __future__ import annotations

import torch
from torch import nn

from forgelm.config import GenerationSettings, ModelConfig
from forgelm.generation.sampling import (
    _top_k_filter,
    _top_p_filter,
    generate,
    sample_next_token,
)
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed

# -- greedy ---------------------------------------------------------------------


def test_greedy_picks_the_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
    config = GenerationSettings(do_sample=False)
    token = sample_next_token(logits, config)
    assert token.item() == 1


def test_greedy_ignores_temperature_and_seed() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
    a = sample_next_token(logits, GenerationSettings(do_sample=False, temperature=10.0, seed=1))
    b = sample_next_token(logits, GenerationSettings(do_sample=False, temperature=0.01, seed=2))
    assert a.item() == b.item() == 1


# -- top-k / top-p filtering ------------------------------------------------------


def test_top_k_filter_keeps_exactly_k_finite_logits() -> None:
    logits = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    filtered = _top_k_filter(logits, top_k=2)
    finite_mask = torch.isfinite(filtered)
    assert finite_mask.sum().item() == 2
    # The two largest values (5.0 at index 0, 4.0 at index 2) survive.
    assert finite_mask[0, 0].item() is True
    assert finite_mask[0, 2].item() is True


def test_top_k_filter_keeps_exactly_k_even_with_a_tie_at_the_cutoff() -> None:
    """Two logits tied for the k-th-largest slot must not both survive --
    FR8's top-k constraint is "exactly top_k candidates", not "at least"."""
    logits = torch.tensor([[5.0, 4.0, 4.0, 1.0]])
    filtered = _top_k_filter(logits, top_k=2)
    finite_mask = torch.isfinite(filtered)
    assert finite_mask.sum().item() == 2
    assert finite_mask[0, 0].item() is True  # the unambiguous top logit


def test_top_k_filter_larger_than_vocab_keeps_everything_finite() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered = _top_k_filter(logits, top_k=100)
    assert torch.isfinite(filtered).all()


def test_top_p_filter_keeps_smallest_nucleus() -> None:
    # Softmax([4, 0, 0, 0]) puts the vast majority of probability mass on
    # index 0 -- a small top_p should keep only that one token.
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    filtered = _top_p_filter(logits, top_p=0.5)
    finite_mask = torch.isfinite(filtered)
    assert finite_mask.sum().item() == 1
    assert finite_mask[0, 0].item() is True


def test_top_p_filter_of_one_keeps_everything() -> None:
    logits = torch.tensor([[4.0, 0.0, -1.0, 2.0]])
    filtered = _top_p_filter(logits, top_p=1.0)
    assert torch.isfinite(filtered).all()


def test_top_p_filter_always_keeps_at_least_the_top_token() -> None:
    """Even an extremely small top_p must not filter out every token --
    otherwise softmax([-inf, -inf, ...]) is NaN and sampling breaks."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered = _top_p_filter(logits, top_p=1e-9)
    assert torch.isfinite(filtered).sum().item() == 1


# -- sampling: reproducibility + constraint satisfaction --------------------------


def test_sampling_is_reproducible_given_the_same_generator_seed() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    config = GenerationSettings(do_sample=True, temperature=1.0, seed=42)

    gen_a = torch.Generator()
    gen_a.manual_seed(config.seed)
    gen_b = torch.Generator()
    gen_b.manual_seed(config.seed)

    token_a = sample_next_token(logits, config, gen_a)
    token_b = sample_next_token(logits, config, gen_b)
    assert token_a.item() == token_b.item()


def test_sampling_with_top_k_only_ever_returns_top_k_tokens() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    config = GenerationSettings(do_sample=True, top_k=2, seed=0)
    generator = torch.Generator()
    generator.manual_seed(0)

    seen = {sample_next_token(logits, config, generator).item() for _ in range(30)}
    assert seen <= {0, 1}  # only the two highest-logit indices


def test_low_temperature_sampling_concentrates_on_the_argmax() -> None:
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.1]])
    config = GenerationSettings(do_sample=True, temperature=0.05, seed=0)
    generator = torch.Generator()
    generator.manual_seed(0)

    draws = [sample_next_token(logits, config, generator).item() for _ in range(20)]
    assert all(d == 2 for d in draws)


# -- generate(): shape, determinism, context-length cropping ----------------------


def _tiny_model() -> tuple[TransformerDecoder, ModelConfig]:
    config = ModelConfig(
        vocab_size=32, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0
    )
    set_seed(0)
    return TransformerDecoder(config), config


def test_generate_returns_prompt_plus_max_new_tokens() -> None:
    model, config = _tiny_model()
    prompt = [1, 2, 3]
    settings = GenerationSettings(max_new_tokens=5, do_sample=False)
    out = generate(model, prompt, settings, context_length=config.context_length)
    assert out[: len(prompt)] == prompt
    assert len(out) == len(prompt) + 5


def test_generate_greedy_is_deterministic() -> None:
    model, config = _tiny_model()
    prompt = [1, 2, 3, 4]
    settings = GenerationSettings(max_new_tokens=6, do_sample=False)
    out_a = generate(model, prompt, settings, context_length=config.context_length)
    out_b = generate(model, prompt, settings, context_length=config.context_length)
    assert out_a == out_b


def test_generate_sampling_is_reproducible_given_the_same_seed() -> None:
    model, config = _tiny_model()
    prompt = [1, 2, 3]
    settings = GenerationSettings(max_new_tokens=8, do_sample=True, temperature=1.2, seed=7)
    out_a = generate(model, prompt, settings, context_length=config.context_length)
    out_b = generate(model, prompt, settings, context_length=config.context_length)
    assert out_a == out_b


def test_generate_crops_to_context_length_without_erroring() -> None:
    """A prompt already at (or generation that grows past) context_length
    must not crash -- generate() must crop its input window every step."""
    model, config = _tiny_model()
    prompt = list(range(config.context_length))  # exactly context_length tokens
    settings = GenerationSettings(max_new_tokens=10, do_sample=False)
    out = generate(model, prompt, settings, context_length=config.context_length)
    assert len(out) == len(prompt) + 10


class _AlwaysPredictsStub(nn.Module):
    """A stub "model" whose forward always assigns the highest logit to a
    fixed token id, regardless of input -- makes "does generate() stop at
    eos_token_id" deterministically testable without depending on what a
    real (randomly initialized) TransformerDecoder happens to predict."""

    def __init__(self, vocab_size: int, always_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.always_id = always_id
        self._unused = nn.Parameter(torch.zeros(1))  # nn.Module needs >=1 param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -10.0)
        logits[..., self.always_id] = 10.0
        return logits


def test_generate_stops_early_at_eos_token() -> None:
    eos_id = 5
    stub = _AlwaysPredictsStub(vocab_size=32, always_id=eos_id)
    prompt = [1, 2, 3]
    settings = GenerationSettings(max_new_tokens=50, do_sample=False)

    out = generate(stub, prompt, settings, context_length=8, eos_token_id=eos_id)

    # The stub always predicts eos_id -- generation must stop after the
    # very first new token, not run all 50 requested steps.
    assert len(out) == len(prompt) + 1
    assert out[-1] == eos_id


def test_generate_restores_model_training_mode() -> None:
    model, config = _tiny_model()
    model.train()
    generate(
        model, [1, 2], GenerationSettings(max_new_tokens=2), context_length=config.context_length
    )
    assert model.training is True

    model.eval()
    generate(
        model, [1, 2], GenerationSettings(max_new_tokens=2), context_length=config.context_length
    )
    assert model.training is False
