"""Library-level dataset-build operations the CLI wraps (also used by tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from forgelm.config import DatasetBuildConfig
from forgelm.dataset import build_dataset, load_text
from forgelm.tokenizer import ByteLevelBPETokenizer


def build_dataset_artifacts(
    config: DatasetBuildConfig, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Build train/val token arrays + stats and write them to ``output_dir``.

    ``output_dir`` overrides ``config.output_dir`` when given (the CLI's
    ``--output`` flag). One of the two must resolve to a real path -- a
    config file alone (``config.output_dir`` set, no CLI override) is
    sufficient to reproduce the full artifact-producing run (FR1).
    """
    resolved_output_dir = output_dir if output_dir is not None else config.output_dir
    if resolved_output_dir is None:
        raise ValueError(
            "no output directory given: pass --output on the CLI or set "
            "'output_dir' in the config file"
        )

    text = load_text(config.input_path)
    tokenizer = ByteLevelBPETokenizer.load(config.tokenizer_path)
    built = build_dataset(text, tokenizer, config.context_length, config.val_fraction)

    output_dir = Path(resolved_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_tokens.npy", built.train_dataset.tokens.numpy())
    np.save(output_dir / "val_tokens.npy", built.val_dataset.tokens.numpy())
    (output_dir / "stats.json").write_text(json.dumps(built.stats.to_dict(), indent=2))

    return {"output_dir": str(output_dir), **built.stats.to_dict()}
