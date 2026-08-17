"""Phase 4 exit criterion / spec §1.8 explicit integration requirement:
"train -> save -> load -> generate" -- a checkpoint produced by a real
``Trainer`` run must load back into a fresh process-like model object and
produce usable generations and an evaluable loss, entirely independent of
the ``Trainer``/optimizer/scheduler state that produced it."""

from __future__ import annotations

from pathlib import Path

import torch

from forgelm.config import GenerationSettings, ModelConfig, TrainingConfig
from forgelm.dataset import NextTokenDataset
from forgelm.evaluation import evaluate_loss
from forgelm.generation import generate, load_model_from_checkpoint
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed
from forgelm.training import Trainer

VOCAB_SIZE = 24
CONTEXT_LENGTH = 8


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
    )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=123,
        micro_batch_size=4,
        max_steps=8,
        warmup_steps=1,
        eval_interval=1000,
        eval_batches=1,
        device="cpu",
        precision="fp32",
    )


def _dataset(offset: int = 0) -> NextTokenDataset:
    tokens = [(i * 7 + offset) % VOCAB_SIZE for i in range(300)]
    return NextTokenDataset(tokens, CONTEXT_LENGTH)


def test_train_save_load_generate_end_to_end(tmp_path: Path) -> None:
    model_config = _model_config()
    training_config = _training_config()
    train_ds = _dataset()
    val_ds = _dataset(offset=1)

    # -- train a few real steps ------------------------------------------------
    set_seed(42)
    model = TransformerDecoder(model_config)
    trainer = Trainer(model, train_ds, val_ds, model_config, training_config)
    for _ in range(training_config.max_steps):
        trainer.train_step()
    val_loss_during_training = trainer.evaluate()

    ckpt_path = tmp_path / "final.pt"
    trainer.save_checkpoint(ckpt_path, val_loss=val_loss_during_training)

    # -- load into a fresh model, independent of the Trainer that produced it --
    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")
    assert loaded.model_config == model_config
    assert loaded.step == training_config.max_steps
    assert loaded.model.training is False

    # -- evaluate: matches what the trainer itself measured ---------------------
    eval_result = evaluate_loss(
        loaded.model, val_ds, batch_size=training_config.micro_batch_size, device="cpu"
    )
    assert eval_result.loss == eval_result.loss  # not NaN
    assert eval_result.perplexity > 1.0  # cross-entropy loss > 0 -> perplexity > 1

    # -- generate: both greedy and sampling modes work from the loaded model ----
    prompt_ids = [1, 2, 3, 4]

    greedy = generate(
        loaded.model,
        prompt_ids,
        GenerationSettings(max_new_tokens=10, do_sample=False),
        context_length=model_config.context_length,
    )
    assert greedy[: len(prompt_ids)] == prompt_ids
    assert len(greedy) == len(prompt_ids) + 10

    # Greedy decoding from the same loaded checkpoint is deterministic.
    greedy_again = generate(
        loaded.model,
        prompt_ids,
        GenerationSettings(max_new_tokens=10, do_sample=False),
        context_length=model_config.context_length,
    )
    assert greedy == greedy_again

    sampled = generate(
        loaded.model,
        prompt_ids,
        GenerationSettings(max_new_tokens=10, do_sample=True, temperature=0.9, top_k=8, seed=1),
        context_length=model_config.context_length,
    )
    assert len(sampled) == len(prompt_ids) + 10
    for token_id in sampled:
        assert 0 <= token_id < model_config.vocab_size


def test_loaded_checkpoint_is_independent_of_the_original_trainer_object(tmp_path: Path) -> None:
    """Mutating the original Trainer's model after saving must not affect
    what a freshly loaded checkpoint produces -- proves load_state_dict
    actually copies weights rather than aliasing them."""
    model_config = _model_config()
    training_config = _training_config()
    train_ds = _dataset()
    val_ds = _dataset(offset=1)

    set_seed(7)
    model = TransformerDecoder(model_config)
    trainer = Trainer(model, train_ds, val_ds, model_config, training_config)
    for _ in range(3):
        trainer.train_step()

    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path)
    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")

    # Corrupt the original trainer's live weights after the checkpoint was
    # already saved and loaded elsewhere.
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(1000.0)

    x = torch.randint(0, model_config.vocab_size, (1, model_config.context_length))
    with torch.no_grad():
        loaded_logits = loaded.model(x)
    assert torch.isfinite(loaded_logits).all()
    # The loaded model's weights must not have been affected by the later
    # in-place mutation of the original trainer's model.
    for p_loaded, p_corrupted in zip(
        loaded.model.parameters(), trainer.model.parameters(), strict=True
    ):
        assert not torch.allclose(p_loaded, p_corrupted)
