"""Unit tests for forgelm.generation.checkpoint_loader (Phase 4 "checkpoint
loader" deliverable)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.generation.checkpoint_loader import load_model_from_checkpoint
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed
from forgelm.training.checkpoint import save_checkpoint
from forgelm.training.optimizer import build_optimizer
from forgelm.training.scheduler import build_scheduler


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=32, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)


def _write_checkpoint(tmp_path: Path, *, step: int = 5, val_loss: float = 2.5) -> Path:
    set_seed(0)
    model_config = _tiny_model_config()
    training_config = TrainingConfig(micro_batch_size=2, max_steps=10, warmup_steps=1)
    model = TransformerDecoder(model_config)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        model_config=model_config,
        training_config=training_config,
        val_loss=val_loss,
    )
    return path


def test_load_model_from_checkpoint_reconstructs_matching_architecture(tmp_path: Path) -> None:
    ckpt_path = _write_checkpoint(tmp_path)
    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")

    assert loaded.model_config == _tiny_model_config()
    assert loaded.step == 5
    assert loaded.val_loss == pytest.approx(2.5)
    assert isinstance(loaded.model, TransformerDecoder)


def test_load_model_from_checkpoint_puts_the_model_in_eval_mode(tmp_path: Path) -> None:
    ckpt_path = _write_checkpoint(tmp_path)
    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")
    assert loaded.model.training is False


def test_load_model_from_checkpoint_restores_exact_weights(tmp_path: Path) -> None:
    set_seed(0)
    model_config = _tiny_model_config()
    training_config = TrainingConfig(micro_batch_size=2, max_steps=10, warmup_steps=1)
    original_model = TransformerDecoder(model_config)
    optimizer = build_optimizer(original_model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=original_model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        model_config=model_config,
        training_config=training_config,
    )

    loaded = load_model_from_checkpoint(path, device="cpu")
    for p_original, p_loaded in zip(
        original_model.parameters(), loaded.model.parameters(), strict=True
    ):
        torch.testing.assert_close(p_original, p_loaded, atol=0.0, rtol=0.0)


def test_load_model_from_checkpoint_produces_a_usable_forward_pass(tmp_path: Path) -> None:
    ckpt_path = _write_checkpoint(tmp_path)
    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")
    x = torch.randint(0, loaded.model_config.vocab_size, (2, 4))
    with torch.no_grad():
        logits = loaded.model(x)
    assert logits.shape == (2, 4, loaded.model_config.vocab_size)
    assert torch.isfinite(logits).all()


def test_load_model_from_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint(tmp_path / "does_not_exist.pt")
