"""Checkpoint format (D7).

A single ``torch.save`` dict containing:

  - ``model_state_dict`` / ``optimizer_state_dict`` / ``scheduler_state_dict``
  - full RNG state (Python ``random``, NumPy, torch CPU + all visible
    CUDA devices)
  - the training ``step`` counter
  - ``model_config`` / ``training_config`` (as plain dicts, for
    inspection without reconstructing dataclasses) and a SHA-256
    ``config_hash`` of both plus (when the caller supplies them, as
    ``forgelm.cli.train_cli.run_training`` always does) the dataset token
    array paths, so a checkpoint can only be resumed against the config
    *and dataset* it was produced with -- a mismatch is reported as a
    clear error instead of ``load_state_dict`` failing deep inside with a
    confusing tensor-shape mismatch, or (worse) silently succeeding while
    training under different hyperparameters, or against a different
    token stream, than intended.

See ``docs/decisions/0009-checkpoint-format.md`` for the full rationale,
including the known limitation that this makes resuming with a
*different* config (e.g. a longer ``max_steps``) an explicit error rather
than a supported "extend training" flow.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from forgelm.config import ModelConfig, TrainingConfig

CHECKPOINT_FORMAT_VERSION = 1


def config_hash(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_paths: tuple[str | Path, str | Path] | None = None,
) -> str:
    """A stable hash of everything that determines a checkpoint's
    shape/semantics *and* the data it was trained against -- used to
    detect resuming with an incompatible config or a different dataset.

    ``dataset_paths`` is ``(train_tokens_path, val_tokens_path)``. It is
    optional (and omitted from the hash when ``None``) because
    ``Trainer`` itself is constructed from already-loaded ``Dataset``
    objects and has no notion of "path" -- only a caller that knows the
    paths (``forgelm.cli.train_cli.run_training``) can supply them. When
    both save and load omit it, the check degrades to the config-only
    comparison; when a caller supplies it, resuming against a checkpoint
    produced from different token-array paths is a hash mismatch, not a
    silent resume onto a different batch stream (see
    ``docs/decisions/0009-checkpoint-format.md``).
    """
    payload: dict[str, Any] = {
        "model": dataclasses.asdict(model_config),
        "training": dataclasses.asdict(training_config),
    }
    if dataset_paths is not None:
        payload["dataset_paths"] = [str(p) for p in dataset_paths]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG this project seeds (see ``forgelm.seed.set_seed``)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    val_loss: float | None = None,
    dataset_paths: tuple[str | Path, str | Path] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "model_config": dataclasses.asdict(model_config),
        "training_config": dataclasses.asdict(training_config),
        "dataset_paths": [str(p) for p in dataset_paths] if dataset_paths is not None else None,
        "config_hash": config_hash(model_config, training_config, dataset_paths),
        "val_loss": val_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load the raw checkpoint payload dict.

    Does not mutate any model/optimizer -- see
    :meth:`forgelm.training.loop.Trainer.load_checkpoint` for the
    higher-level "resume a Trainer" entry point that also restores state
    into live objects.

    ``weights_only=False``: this payload intentionally contains more than
    tensors (RNG state tuples, plain dicts/floats/ints), which torch's
    default ``weights_only=True`` unpickling would reject. This is safe
    here because checkpoints are only ever written by this project's own
    :func:`save_checkpoint` and loaded from trusted local paths (D3:
    "prefer deterministic local development" -- no untrusted or networked
    checkpoint loading is in scope for this project).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return payload


def verify_config_hash(
    payload: dict[str, Any],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_paths: tuple[str | Path, str | Path] | None = None,
) -> None:
    """Raise a clear error if ``payload`` was not produced by the given
    configs (and, when ``dataset_paths`` is supplied, the given dataset)."""
    expected = config_hash(model_config, training_config, dataset_paths)
    actual = payload.get("config_hash")
    if actual != expected:
        raise ValueError(
            "checkpoint config_hash does not match the given model/training config "
            "(and/or dataset paths) -- resuming would silently train with different "
            "hyperparameters/architecture, or against a different token stream, than "
            f"the checkpoint was produced with (expected {expected}, got {actual})"
        )
