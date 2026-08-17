from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.training.checkpoint import (
    capture_rng_state,
    config_hash,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    verify_config_hash,
)
from forgelm.training.optimizer import build_optimizer
from forgelm.training.scheduler import build_scheduler


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=32, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)


def _tiny_training_config() -> TrainingConfig:
    return TrainingConfig(micro_batch_size=2, max_steps=10, warmup_steps=2)


# -- config_hash --------------------------------------------------------------------


def test_config_hash_is_deterministic() -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    assert config_hash(model_config, training_config) == config_hash(model_config, training_config)


def test_config_hash_changes_when_model_config_changes() -> None:
    training_config = _tiny_training_config()
    model_config_a = _tiny_model_config()
    model_config_b = dataclasses.replace(model_config_a, d_model=32)
    hash_a = config_hash(model_config_a, training_config)
    hash_b = config_hash(model_config_b, training_config)
    assert hash_a != hash_b


def test_config_hash_changes_when_training_config_changes() -> None:
    model_config = _tiny_model_config()
    hash_a = config_hash(model_config, TrainingConfig(lr=1e-3))
    hash_b = config_hash(model_config, TrainingConfig(lr=5e-4))
    assert hash_a != hash_b


def test_config_hash_changes_when_dataset_paths_change() -> None:
    """Resuming against a different corpus/tokenizer run -- a different
    train_tokens_path/val_tokens_path -- must not silently hash the same
    as the original dataset, even though ModelConfig/TrainingConfig are
    unchanged."""
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    hash_a = config_hash(model_config, training_config, ("train_a.npy", "val_a.npy"))
    hash_b = config_hash(model_config, training_config, ("train_b.npy", "val_b.npy"))
    assert hash_a != hash_b


def test_config_hash_with_and_without_dataset_paths_differ() -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    hash_no_paths = config_hash(model_config, training_config)
    hash_with_paths = config_hash(model_config, training_config, ("train.npy", "val.npy"))
    assert hash_no_paths != hash_with_paths


# -- RNG state round trip -------------------------------------------------------------


def test_rng_state_round_trip_reproduces_subsequent_draws() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state = capture_rng_state()
    draw_a = (random.random(), np.random.rand(), torch.rand(3).tolist())

    # Advance all three RNGs arbitrarily, then restore and redraw.
    random.random()
    np.random.rand()
    torch.rand(5)

    restore_rng_state(state)
    draw_b = (random.random(), np.random.rand(), torch.rand(3).tolist())

    assert draw_a == draw_b


# -- save / load round trip ----------------------------------------------------------


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    model = nn.Linear(model_config.d_model, model_config.d_model)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    # Take one optimizer step so optimizer/scheduler state is non-trivial.
    loss = model(torch.randn(2, model_config.d_model)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "nested" / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=7,
        model_config=model_config,
        training_config=training_config,
        val_loss=1.2345,
    )
    assert path.exists()

    payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert payload["val_loss"] == pytest.approx(1.2345)
    assert payload["config_hash"] == config_hash(model_config, training_config)
    assert set(payload["model_state_dict"].keys()) == set(model.state_dict().keys())
    assert "state" in payload["optimizer_state_dict"]  # AdamW step/exp_avg buffers
    assert "torch" in payload["rng_state"]

    # Loading into a fresh model reproduces identical parameters.
    fresh_model = nn.Linear(model_config.d_model, model_config.d_model)
    fresh_model.load_state_dict(payload["model_state_dict"])
    for p_a, p_b in zip(model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(p_a, p_b)


def test_load_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_save_checkpoint_records_dataset_paths_in_hash(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    model = nn.Linear(model_config.d_model, model_config.d_model)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)
    dataset_paths = (str(tmp_path / "train.npy"), str(tmp_path / "val.npy"))

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        model_config=model_config,
        training_config=training_config,
        dataset_paths=dataset_paths,
    )

    payload = load_checkpoint(path)
    assert payload["dataset_paths"] == [str(p) for p in dataset_paths]
    assert payload["config_hash"] == config_hash(model_config, training_config, dataset_paths)
    # Verifying without the matching dataset_paths must fail -- proves the
    # paths actually participate in the hash, not just get recorded inertly.
    with pytest.raises(ValueError, match="config_hash"):
        verify_config_hash(payload, model_config, training_config)


# -- verify_config_hash ---------------------------------------------------------------


def test_verify_config_hash_accepts_matching_config() -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    payload = {"config_hash": config_hash(model_config, training_config)}
    verify_config_hash(payload, model_config, training_config)  # must not raise


def test_verify_config_hash_raises_on_mismatch() -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    payload = {"config_hash": "not-the-real-hash"}
    with pytest.raises(ValueError, match="config_hash"):
        verify_config_hash(payload, model_config, training_config)


def test_verify_config_hash_raises_when_dataset_paths_differ() -> None:
    """The exact scenario a silent-resume bug would allow: same model and
    training config, but the checkpoint was produced from a different
    dataset (e.g. a regenerated/repointed corpus) -- must be a loud
    error, not a silent resume onto a different token stream."""
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    payload = {
        "config_hash": config_hash(
            model_config, training_config, ("original_train.npy", "original_val.npy")
        )
    }
    with pytest.raises(ValueError, match="config_hash"):
        verify_config_hash(
            payload, model_config, training_config, ("different_train.npy", "different_val.npy")
        )


def test_verify_config_hash_accepts_matching_dataset_paths() -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    paths = ("train.npy", "val.npy")
    payload = {"config_hash": config_hash(model_config, training_config, paths)}
    verify_config_hash(payload, model_config, training_config, paths)  # must not raise
