"""Load a trained model from a checkpoint for inference (Phase 4's
"checkpoint loader" deliverable).

Deliberately independent of ``forgelm.training.loop.Trainer``: generation
and evaluation need a model in eval mode on a device, nothing else
(no optimizer, no scheduler, no batch-stream position) -- reconstructing a
full ``Trainer`` just to read ``trainer.model`` would pull in optimizer/
scheduler construction the caller doesn't need and couples this module to
``forgelm.training`` internals it has no business depending on (see the
module boundary table in ``docs/architecture.md``: ``forgelm.generation``
"must not contain" training logic).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from forgelm.config import ModelConfig, TrainingConfig, config_from_dict
from forgelm.model import TransformerDecoder
from forgelm.training.checkpoint import load_checkpoint


@dataclasses.dataclass
class LoadedCheckpoint:
    """Everything :func:`load_model_from_checkpoint` reconstructs from a
    checkpoint file, ready for generation/evaluation."""

    model: TransformerDecoder
    model_config: ModelConfig
    training_config: TrainingConfig
    step: int
    val_loss: float | None
    checkpoint_path: str


def load_model_from_checkpoint(path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
    """Reconstruct a :class:`~forgelm.model.TransformerDecoder` from a
    checkpoint's ``model_config`` + ``model_state_dict``, in eval mode on
    ``device``.

    Raises whatever :func:`forgelm.training.checkpoint.load_checkpoint`
    raises (``FileNotFoundError`` for a missing path) plus a ``KeyError``
    if the payload is missing an expected key -- both surface as clear
    CLI errors rather than a deep ``load_state_dict`` shape mismatch.
    """
    payload: dict[str, Any] = load_checkpoint(path, map_location=device)
    model_config = config_from_dict(payload["model_config"], ModelConfig)
    training_config = config_from_dict(payload["training_config"], TrainingConfig)

    model = TransformerDecoder(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        model_config=model_config,
        training_config=training_config,
        step=payload["step"],
        val_loss=payload.get("val_loss"),
        checkpoint_path=str(path),
    )
