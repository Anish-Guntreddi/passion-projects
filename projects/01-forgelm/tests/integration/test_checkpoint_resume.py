"""Phase 3 exit criterion: "Interrupted training resumes with a documented
equivalent trajectory."

The core proof: an uninterrupted N-step training run and a run
interrupted at step K < N (checkpointed, then resumed into a *brand new*
model/optimizer/scheduler object with different pre-load weights,
simulating a real process restart) must produce matching per-step losses
for steps K..N, within a numerical tolerance (never exact float equality,
per the project-wide rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.dataset import NextTokenDataset, iter_batches
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


def _training_config(max_steps: int, seed: int = 123) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        micro_batch_size=4,
        grad_accum_steps=1,
        max_steps=max_steps,
        warmup_steps=2,
        lr=1e-3,
        eval_interval=1000,  # not exercised in this test
        eval_batches=1,
        device="cpu",
        precision="fp32",
    )


def _make_dataset(offset: int = 0) -> NextTokenDataset:
    tokens = [(i * 7 + offset) % VOCAB_SIZE for i in range(400)]
    return NextTokenDataset(tokens, CONTEXT_LENGTH)


def _build_trainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    init_seed: int,
) -> Trainer:
    set_seed(init_seed)
    model = TransformerDecoder(model_config)
    return Trainer(model, train_ds, val_ds, model_config, training_config)


def test_trainer_resume_reproduces_equivalent_loss_trajectory(tmp_path: Path) -> None:
    model_config = _model_config()
    total_steps = 12
    resume_at = 5
    training_config = _training_config(max_steps=total_steps)

    train_ds = _make_dataset()
    val_ds = _make_dataset(offset=1)

    # -- reference: one uninterrupted run for all `total_steps` --------------------
    reference_trainer = _build_trainer(
        model_config, training_config, train_ds, val_ds, init_seed=42
    )
    reference_losses = [reference_trainer.train_step().loss for _ in range(total_steps)]

    # -- interrupted run: train `resume_at` steps (same init seed -> identical
    # initial weights and identical batch stream up to this point), then
    # checkpoint --------------------------------------------------------------------
    trainer_a = _build_trainer(model_config, training_config, train_ds, val_ds, init_seed=42)
    losses_before_interrupt = [trainer_a.train_step().loss for _ in range(resume_at)]
    ckpt_path = tmp_path / "interrupted.pt"
    trainer_a.save_checkpoint(ckpt_path)

    # -- simulate a real process restart: a brand new model constructed from a
    # *different* seed, so its weights differ from the checkpoint. Resuming
    # must overwrite them, not merely happen to agree -------------------------------
    set_seed(999)
    fresh_model = TransformerDecoder(model_config)
    assert any(
        not torch.equal(p_a, p_b)
        for p_a, p_b in zip(fresh_model.parameters(), trainer_a.model.parameters(), strict=True)
    ), "sanity check: pre-resume weights must differ from the checkpointed weights"

    trainer_b = Trainer.load_checkpoint(
        ckpt_path, fresh_model, train_ds, val_ds, model_config, training_config
    )
    assert trainer_b.step == resume_at
    # Loading the checkpoint must have overwritten fresh_model's weights to
    # match what was saved.
    for p_saved, p_loaded in zip(
        trainer_a.model.parameters(), trainer_b.model.parameters(), strict=True
    ):
        torch.testing.assert_close(p_saved, p_loaded, atol=0.0, rtol=0.0)

    losses_after_resume = [trainer_b.train_step().loss for _ in range(total_steps - resume_at)]

    resumed_losses = losses_before_interrupt + losses_after_resume
    assert len(resumed_losses) == len(reference_losses)
    for step_idx, (ref, resumed) in enumerate(zip(reference_losses, resumed_losses, strict=True)):
        assert resumed == pytest.approx(ref, rel=1e-4, abs=1e-6), (
            f"loss mismatch at step {step_idx}"
        )


def test_trainer_load_checkpoint_restores_optimizer_and_scheduler_state(tmp_path: Path) -> None:
    model_config = _model_config()
    training_config = _training_config(max_steps=10)
    train_ds = _make_dataset()
    val_ds = _make_dataset(offset=1)

    trainer = _build_trainer(model_config, training_config, train_ds, val_ds, init_seed=7)
    for _ in range(4):
        trainer.train_step()
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path)

    saved_lr = trainer.optimizer.param_groups[0]["lr"]
    saved_scheduler_state = trainer.scheduler.state_dict()

    set_seed(0)
    fresh_model = TransformerDecoder(model_config)
    resumed = Trainer.load_checkpoint(
        ckpt_path, fresh_model, train_ds, val_ds, model_config, training_config
    )

    assert resumed.step == 4
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
    assert resumed.scheduler.state_dict()["last_epoch"] == saved_scheduler_state["last_epoch"]
    # AdamW's per-parameter step count in its restored state must match too.
    original_steps = [s["step"].item() for s in trainer.optimizer.state.values()]
    resumed_steps = [s["step"].item() for s in resumed.optimizer.state.values()]
    assert resumed_steps == original_steps


def test_resume_batch_stream_continues_from_the_correct_position() -> None:
    """A focused check of the skip-based resume mechanism itself (see
    docs/decisions/0009-checkpoint-format.md), independent of model
    weights: the first batch a resumed iterator yields must equal the
    batch the uninterrupted stream would have yielded next at that
    position."""
    train_ds = _make_dataset()
    seed = 55
    batch_size = 4
    already_consumed = 6

    reference_iter = iter_batches(train_ds, batch_size, seed=seed)
    for _ in range(already_consumed):
        next(reference_iter)
    expected_x, expected_y = next(reference_iter)

    resumed_iter = iter_batches(train_ds, batch_size, seed=seed, skip_batches=already_consumed)
    actual_x, actual_y = next(resumed_iter)

    assert torch.equal(expected_x, actual_x)
    assert torch.equal(expected_y, actual_y)
