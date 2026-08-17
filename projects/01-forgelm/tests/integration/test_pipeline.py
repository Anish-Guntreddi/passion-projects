"""Phase 1 exit criterion: text -> deterministic training batches, reconstructable.

Exercises the real example corpus (examples/alice_in_wonderland.txt), not
just a synthetic string, and goes through the same library entry points the
CLI uses (forgelm.cli.tokenizer_cli / forgelm.cli.dataset_cli).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forgelm.cli.dataset_cli import build_dataset_artifacts
from forgelm.cli.tokenizer_cli import train_tokenizer
from forgelm.config import DatasetBuildConfig, TokenizerTrainConfig
from forgelm.dataset import build_dataset, make_dataloader
from forgelm.tokenizer import ByteLevelBPETokenizer, compute_compression_stats


def test_full_pipeline_on_example_corpus_reconstructs_exactly(
    example_corpus_text: str, tmp_path: Path
) -> None:
    tokenizer = ByteLevelBPETokenizer(vocab_size=600)
    tokenizer.train(example_corpus_text)
    assert tokenizer.num_merges > 0

    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(tok_path)
    reloaded = ByteLevelBPETokenizer.load(tok_path)
    assert reloaded.merges == tokenizer.merges

    stats = compute_compression_stats(reloaded, example_corpus_text)
    assert stats["bytes_per_token"] > 1.0  # BPE beats raw byte-per-token baseline

    built = build_dataset(example_corpus_text, reloaded, context_length=64, val_fraction=0.1)
    assert built.stats.train_tokens + built.stats.val_tokens == built.stats.num_tokens

    loader = make_dataloader(built.train_dataset, batch_size=8, seed=0)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (8, 64)
    assert y_batch.shape == (8, 64)
    assert torch.equal(x_batch[:, 1:], y_batch[:, :-1])

    # Decode the concatenated *id lists* in one call, not the concatenation
    # of two separately-decoded strings: the example corpus contains
    # multi-byte UTF-8 characters (curly quotes etc.), and a learned merge
    # can legally group bytes across a character boundary, so decoding an
    # interior slice in isolation is not guaranteed to be valid UTF-8 (see
    # docs/decisions/0002-tokenizer-scope.md's "known limitation"). This is
    # exactly the operation that IS always guaranteed exact.
    all_ids = built.train_dataset.tokens.tolist() + built.val_dataset.tokens.tolist()
    assert reloaded.decode(all_ids) == example_corpus_text


def test_cli_wrapper_functions_train_and_build(example_corpus_text: str, tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(example_corpus_text[:20_000], encoding="utf-8")

    tok_config = TokenizerTrainConfig(input_path=input_path, vocab_size=400)
    tok_output = tmp_path / "tokenizer.json"
    train_result = train_tokenizer(tok_config, tok_output)
    assert tok_output.exists()
    assert train_result["num_merges"] > 0
    assert train_result["vocab_size"] == 400

    ds_config = DatasetBuildConfig(
        input_path=input_path, tokenizer_path=tok_output, context_length=32, val_fraction=0.15
    )
    out_dir = tmp_path / "built"
    build_result = build_dataset_artifacts(ds_config, out_dir)
    assert (out_dir / "train_tokens.npy").exists()
    assert (out_dir / "val_tokens.npy").exists()
    assert (out_dir / "stats.json").exists()
    assert build_result["train_tokens"] + build_result["val_tokens"] == build_result["num_tokens"]

    train_arr = np.load(out_dir / "train_tokens.npy")
    val_arr = np.load(out_dir / "val_tokens.npy")
    tokenizer = ByteLevelBPETokenizer.load(tok_output)
    # See the full-sequence-decode note in test_full_pipeline_on_example_corpus... above.
    reconstructed = tokenizer.decode(train_arr.tolist() + val_arr.tolist())
    assert reconstructed == input_path.read_text(encoding="utf-8")


def test_train_tokenizer_reproduces_full_run_from_config_alone(
    example_corpus_text: str, tmp_path: Path
) -> None:
    """FR1: a single config file (no CLI/library --output override) must be
    sufficient to reproduce the full artifact-producing run."""
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(example_corpus_text[:5_000], encoding="utf-8")
    output_path = tmp_path / "tok.json"

    config = TokenizerTrainConfig(input_path=input_path, output_path=output_path, vocab_size=300)
    result = train_tokenizer(config)  # no output_path argument at all

    assert output_path.exists()
    assert result["output_path"] == str(output_path)


def test_train_tokenizer_without_any_output_raises_clear_error(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello world", encoding="utf-8")
    config = TokenizerTrainConfig(input_path=input_path, vocab_size=300)  # no output_path
    with pytest.raises(ValueError, match="output"):
        train_tokenizer(config)


def test_build_dataset_artifacts_reproduces_full_run_from_config_alone(
    example_corpus_text: str, tmp_path: Path
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(example_corpus_text[:5_000], encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    train_tokenizer(TokenizerTrainConfig(input_path=input_path, vocab_size=300), tok_path)

    output_dir = tmp_path / "built"
    config = DatasetBuildConfig(
        input_path=input_path, tokenizer_path=tok_path, output_dir=output_dir, context_length=16
    )
    result = build_dataset_artifacts(config)  # no output_dir argument at all

    assert (output_dir / "train_tokens.npy").exists()
    assert result["output_dir"] == str(output_dir)


def test_build_dataset_artifacts_without_any_output_raises_clear_error(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello world " * 20, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    train_tokenizer(TokenizerTrainConfig(input_path=input_path, vocab_size=300), tok_path)

    config = DatasetBuildConfig(input_path=input_path, tokenizer_path=tok_path)  # no output_dir
    with pytest.raises(ValueError, match="output"):
        build_dataset_artifacts(config)


def test_tokenizer_training_is_deterministic_on_example_corpus(example_corpus_text: str) -> None:
    tok_a = ByteLevelBPETokenizer(vocab_size=500)
    tok_a.train(example_corpus_text)
    tok_b = ByteLevelBPETokenizer(vocab_size=500)
    tok_b.train(example_corpus_text)
    assert tok_a.merges == tok_b.merges
