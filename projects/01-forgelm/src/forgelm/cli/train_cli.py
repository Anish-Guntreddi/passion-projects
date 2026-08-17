"""Library-level training operations the CLI wraps (also used by tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from forgelm.config import TrainConfig
from forgelm.dataset import NextTokenDataset
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed
from forgelm.training import MetricsLogger, Trainer


def _load_token_dataset(path: str | Path, context_length: int) -> NextTokenDataset:
    tokens = np.load(path)
    return NextTokenDataset(tokens.tolist(), context_length)


def run_training(config: TrainConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run (or resume) a training run end to end and write checkpoints.

    ``output_dir`` overrides ``config.output_dir`` when given (the CLI's
    ``--output`` flag), matching the ``tokenizer train`` / ``dataset
    build`` commands' convention (FR1: a config file alone is sufficient
    to reproduce the full run).
    """
    resolved_output_dir = output_dir if output_dir is not None else config.output_dir
    if resolved_output_dir is None:
        raise ValueError(
            "no output directory given: pass --output on the CLI or set "
            "'output_dir' in the config file"
        )
    resolved_output_dir = Path(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_token_dataset(config.train_tokens_path, config.model.context_length)
    val_dataset = _load_token_dataset(config.val_tokens_path, config.model.context_length)
    # Folded into the checkpoint's config_hash (see
    # forgelm.training.checkpoint.config_hash) so resuming against a
    # *different* dataset than the checkpoint was produced with is a
    # loud, immediate error instead of a silent resume onto a different
    # token stream.
    dataset_paths = (str(config.train_tokens_path), str(config.val_tokens_path))

    # Seed before constructing the model so weight init is reproducible
    # given (training_config.seed, model architecture). When resuming, the
    # freshly-initialized weights below are immediately overwritten by the
    # checkpoint's state_dict inside Trainer.load_checkpoint -- construction
    # still needs *a* model instance of the right architecture to load into.
    set_seed(config.training.seed)
    model = TransformerDecoder(config.model)

    if config.resume_from is not None:
        trainer = Trainer.load_checkpoint(
            config.resume_from,
            model,
            train_dataset,
            val_dataset,
            config.model,
            config.training,
            dataset_paths=dataset_paths,
        )
    else:
        trainer = Trainer(
            model, train_dataset, val_dataset, config.model, config.training, dataset_paths
        )

    # FR6 "logging": persist the full loss/lr/grad_norm/val_loss
    # trajectory of this run as JSON lines, so it can be reloaded and
    # plotted after the fact -- not just the single final-state summary
    # this function returns. See forgelm.training.metrics_logger.
    metrics_logger = MetricsLogger(resolved_output_dir / "metrics.jsonl")

    start_step = trainer.step
    last_val_loss: float | None = None
    final_checkpoint_path = resolved_output_dir / "final.pt"

    while trainer.step < config.training.max_steps:
        metrics = trainer.train_step()
        metrics_logger.log_step(metrics)
        is_last_step = metrics.step >= config.training.max_steps
        if metrics.step % config.training.eval_interval == 0 or is_last_step:
            last_val_loss = trainer.evaluate()
            metrics_logger.log_eval(metrics.step, last_val_loss)
        if (
            config.checkpoint_interval
            and metrics.step % config.checkpoint_interval == 0
            and not is_last_step
        ):
            trainer.save_checkpoint(
                resolved_output_dir / f"step_{metrics.step}.pt", val_loss=last_val_loss
            )

    if last_val_loss is None:
        last_val_loss = trainer.evaluate()
        metrics_logger.log_eval(trainer.step, last_val_loss)
    trainer.save_checkpoint(final_checkpoint_path, val_loss=last_val_loss)

    return {
        "output_dir": str(resolved_output_dir),
        "start_step": start_step,
        "final_step": trainer.step,
        "final_val_loss": last_val_loss,
        "checkpoint_path": str(final_checkpoint_path),
    }
