"""Training loop orchestration.

The single place FR6/FR7 come together: AdamW + LR schedule + gradient
clipping + gradient accumulation + mixed precision + periodic validation
+ checkpoint save/resume. Owns *how to run steps*, not model architecture
(``forgelm.model``) or how raw text becomes token arrays
(``forgelm.dataset``) -- see the module boundary table in
``docs/architecture.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.dataset import iter_batches
from forgelm.training.checkpoint import (
    load_checkpoint as _load_checkpoint_payload,
)
from forgelm.training.checkpoint import (
    restore_rng_state,
    save_checkpoint,
    verify_config_hash,
)
from forgelm.training.optimizer import build_optimizer
from forgelm.training.precision import PrecisionPolicy, resolve_precision
from forgelm.training.scheduler import build_scheduler


@dataclasses.dataclass(frozen=True)
class StepMetrics:
    """What :meth:`Trainer.train_step` reports back for one optimizer step."""

    step: int
    loss: float
    lr: float
    grad_norm: float


class Trainer:
    """Orchestrates one model's training run against pre-built token datasets."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        dataset_paths: tuple[str | Path, str | Path] | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_config = model_config
        self.training_config = training_config
        # (train_tokens_path, val_tokens_path), when the caller knows them
        # (forgelm.cli.train_cli.run_training always does) -- folded into
        # the checkpoint's config_hash so resuming against a *different*
        # dataset than the checkpoint was produced with is a loud error
        # instead of a silent resume onto a different token stream. See
        # forgelm.training.checkpoint.config_hash.
        self.dataset_paths = dataset_paths

        self.device = torch.device(training_config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TrainingConfig.device='cuda' but no CUDA device is available")
        self.model.to(self.device)

        self.optimizer = build_optimizer(self.model, training_config)
        self.scheduler = build_scheduler(self.optimizer, training_config)
        self.precision: PrecisionPolicy = resolve_precision(self.device, training_config.precision)

        self.step = 0
        self._batch_iter = iter_batches(
            train_dataset, training_config.micro_batch_size, seed=training_config.seed
        )

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.precision.device_type,
            dtype=self.precision.autocast_dtype,
            enabled=self.precision.enabled,
        )

    # -- one optimizer step (accumulated over grad_accum_steps micro-batches) --

    def train_step(self) -> StepMetrics:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum_steps = self.training_config.grad_accum_steps
        accum_loss = 0.0

        for _ in range(accum_steps):
            x, y = next(self._batch_iter)
            x, y = x.to(self.device), y.to(self.device)
            with self._autocast():
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / accum_steps).backward()
            accum_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.training_config.grad_clip_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        return StepMetrics(
            step=self.step,
            loss=accum_loss / accum_steps,
            lr=self.optimizer.param_groups[0]["lr"],
            grad_norm=float(grad_norm),
        )

    # -- validation (FR9's "deterministic eval mode"; full perplexity /
    # sample-report formatting is Phase 4's job, forgelm.evaluation) --

    @torch.no_grad()
    def evaluate(self, num_batches: int | None = None) -> float:
        """Average cross-entropy loss over ``num_batches`` validation
        batches (default: ``training_config.eval_batches``), in eval mode
        with gradients disabled. Uses a fixed seed independent of the
        training batch stream, so repeated calls at the same point in
        training see the same validation batches."""
        num_batches = num_batches if num_batches is not None else self.training_config.eval_batches
        self.model.eval()
        eval_iter = iter_batches(self.val_dataset, self.training_config.micro_batch_size, seed=0)
        total_loss = 0.0
        for _ in range(num_batches):
            x, y = next(eval_iter)
            x, y = x.to(self.device), y.to(self.device)
            with self._autocast():
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
        self.model.train()
        return total_loss / num_batches

    # -- checkpointing --

    def save_checkpoint(self, path: str | Path, *, val_loss: float | None = None) -> None:
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            model_config=self.model_config,
            training_config=self.training_config,
            val_loss=val_loss,
            dataset_paths=self.dataset_paths,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        dataset_paths: tuple[str | Path, str | Path] | None = None,
    ) -> Trainer:
        """Reconstruct a :class:`Trainer` from a checkpoint.

        ``model`` should be a freshly-constructed (untrained) instance of
        the same architecture ``model_config`` describes -- its weights
        are immediately overwritten by the checkpoint's
        ``model_state_dict``, so the caller does not need to seed RNG
        carefully before constructing it. The batch stream and RNG state
        *are* restored precisely, which is what makes the resumed run's
        trajectory match the uninterrupted run's (Phase 3 exit criterion).

        ``dataset_paths`` should be the same ``(train_tokens_path,
        val_tokens_path)`` the checkpoint was saved with -- passing it
        makes :func:`~forgelm.training.checkpoint.verify_config_hash` also
        catch resuming against a different dataset than the one the
        checkpoint was produced from, not just a different model/training
        config.
        """
        payload = _load_checkpoint_payload(path)
        verify_config_hash(payload, model_config, training_config, dataset_paths)

        trainer = cls(
            model, train_dataset, val_dataset, model_config, training_config, dataset_paths
        )
        trainer.model.load_state_dict(payload["model_state_dict"])
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        trainer.step = payload["step"]

        # Re-derive the exact batch stream the checkpointed run was on:
        # iter_batches is a pure function of (dataset, batch_size, seed),
        # so skipping the number of micro-batches already consumed
        # reproduces the same upcoming batch order the uninterrupted run
        # would have seen from this point on.
        already_consumed = trainer.step * training_config.grad_accum_steps
        trainer._batch_iter = iter_batches(
            train_dataset,
            training_config.micro_batch_size,
            seed=training_config.seed,
            skip_batches=already_consumed,
        )
        return trainer
