"""Library-level tokenizer operations the CLI wraps (also used directly by tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forgelm.config import TokenizerTrainConfig
from forgelm.dataset import load_text
from forgelm.tokenizer import ByteLevelBPETokenizer, compute_compression_stats


def train_tokenizer(
    config: TokenizerTrainConfig, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Train and save a tokenizer.

    ``output_path`` overrides ``config.output_path`` when given (the CLI's
    ``--output`` flag). One of the two must resolve to a real path -- a
    config file alone (``config.output_path`` set, no CLI override) is
    sufficient to reproduce the full artifact-producing run (FR1).
    """
    resolved_output = output_path if output_path is not None else config.output_path
    if resolved_output is None:
        raise ValueError(
            "no output path given: pass --output on the CLI or set 'output_path' in the config file"
        )

    text = load_text(config.input_path)
    tokenizer = ByteLevelBPETokenizer(
        vocab_size=config.vocab_size, special_tokens=config.special_tokens
    )
    tokenizer.train(text)
    tokenizer.save(resolved_output)
    stats = compute_compression_stats(tokenizer, text)
    return {
        "output_path": str(resolved_output),
        "vocab_size": tokenizer.vocab_size,
        "num_merges": tokenizer.num_merges,
        **stats,
    }


def encode_text(tokenizer_path: str | Path, text: str) -> list[int]:
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    return tokenizer.encode(text)


def decode_ids(tokenizer_path: str | Path, ids: list[int]) -> str:
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    return tokenizer.decode(ids)
