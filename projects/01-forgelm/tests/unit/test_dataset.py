from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forgelm.dataset import (
    NextTokenDataset,
    build_dataset,
    compute_dataset_stats,
    iter_batches,
    load_text,
    make_dataloader,
    tokenize_corpus,
    train_val_split,
)
from forgelm.tokenizer import ByteLevelBPETokenizer


@pytest.fixture()
def trained_tokenizer(tiny_corpus: str) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    return tok


# -- load_text ------------------------------------------------------------------


def test_load_text_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("hello world", encoding="utf-8")
    assert load_text(path) == "hello world"


def test_load_text_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_text("does/not/exist.txt")


# -- train_val_split --------------------------------------------------------------


def test_train_val_split_is_contiguous_and_covers_all_tokens() -> None:
    tokens = list(range(100))
    train, val = train_val_split(tokens, val_fraction=0.2)
    assert train == list(range(80))
    assert val == list(range(80, 100))
    assert train + val == tokens


def test_train_val_split_is_deterministic() -> None:
    tokens = list(range(57))
    a = train_val_split(tokens, val_fraction=0.15)
    b = train_val_split(tokens, val_fraction=0.15)
    assert a == b


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.2])
def test_train_val_split_rejects_bad_fraction(val_fraction: float) -> None:
    with pytest.raises(ValueError):
        train_val_split(list(range(10)), val_fraction=val_fraction)


# -- NextTokenDataset -------------------------------------------------------------


def test_next_token_dataset_shapes_and_shift() -> None:
    tokens = list(range(20))
    ds = NextTokenDataset(tokens, context_length=5)
    assert len(ds) == 15
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]
    assert torch.equal(x[1:], y[:-1])  # y is x shifted by one position


def test_next_token_dataset_last_index() -> None:
    tokens = list(range(20))
    ds = NextTokenDataset(tokens, context_length=5)
    x, y = ds[len(ds) - 1]
    assert x.tolist() == [14, 15, 16, 17, 18]
    assert y.tolist() == [15, 16, 17, 18, 19]


def test_next_token_dataset_out_of_range_raises() -> None:
    ds = NextTokenDataset(list(range(10)), context_length=3)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_next_token_dataset_rejects_too_few_tokens() -> None:
    with pytest.raises(ValueError):
        NextTokenDataset(list(range(3)), context_length=5)


def test_next_token_dataset_rejects_non_positive_context_length() -> None:
    with pytest.raises(ValueError):
        NextTokenDataset(list(range(10)), context_length=0)


# -- deterministic dataloader -------------------------------------------------------


def test_make_dataloader_same_seed_same_batch_order() -> None:
    ds = NextTokenDataset(list(range(200)), context_length=8)
    loader_a = make_dataloader(ds, batch_size=4, seed=0)
    loader_b = make_dataloader(ds, batch_size=4, seed=0)
    xa, ya = next(iter(loader_a))
    xb, yb = next(iter(loader_b))
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_make_dataloader_different_seed_can_differ_order() -> None:
    ds = NextTokenDataset(list(range(500)), context_length=8)
    loader_a = make_dataloader(ds, batch_size=4, seed=0)
    loader_b = make_dataloader(ds, batch_size=4, seed=1)
    xa, _ = next(iter(loader_a))
    xb, _ = next(iter(loader_b))
    assert not torch.equal(xa, xb)


# -- dataset statistics ------------------------------------------------------------


def test_compute_dataset_stats(tiny_corpus: str, trained_tokenizer: ByteLevelBPETokenizer) -> None:
    tokens = tokenize_corpus(tiny_corpus, trained_tokenizer)
    stats = compute_dataset_stats(tiny_corpus, tokens, val_fraction=0.25)
    assert stats.num_tokens == len(tokens)
    assert stats.train_tokens + stats.val_tokens == stats.num_tokens
    assert stats.unique_tokens <= stats.num_tokens
    assert stats.num_bytes == len(tiny_corpus.encode("utf-8"))


# -- end-to-end reconstructability (Phase 1 exit criterion) ------------------------


def test_build_dataset_reconstructs_original_text(
    tiny_corpus: str, trained_tokenizer: ByteLevelBPETokenizer
) -> None:
    built = build_dataset(tiny_corpus, trained_tokenizer, context_length=6, val_fraction=0.2)

    train_tokens = built.train_dataset.tokens.tolist()
    val_tokens = built.val_dataset.tokens.tolist()

    # Reconstructing train + val tokens must exactly reproduce the source
    # text: the split is a pure partition of the token stream and
    # byte-level BPE decode() is lossless. Decode the concatenated id list
    # in one call (always safe) rather than concatenating two separately
    # decoded strings (only safe for single-byte-per-character corpora --
    # see docs/decisions/0002-tokenizer-scope.md's "known limitation").
    reconstructed = trained_tokenizer.decode(train_tokens + val_tokens)
    assert reconstructed == tiny_corpus


def test_iter_batches_is_deterministic_given_seed() -> None:
    ds = NextTokenDataset(list(range(300)), context_length=8)
    it_a = iter_batches(ds, batch_size=4, seed=1)
    it_b = iter_batches(ds, batch_size=4, seed=1)
    for _ in range(10):
        xa, ya = next(it_a)
        xb, yb = next(it_b)
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)


def test_iter_batches_different_seed_differs() -> None:
    ds = NextTokenDataset(list(range(300)), context_length=8)
    it_a = iter_batches(ds, batch_size=4, seed=1)
    it_b = iter_batches(ds, batch_size=4, seed=2)
    xa, _ = next(it_a)
    xb, _ = next(it_b)
    assert not torch.equal(xa, xb)


def test_iter_batches_is_infinite_across_epoch_boundaries() -> None:
    # A tiny dataset with few windows per epoch -- pull more batches than
    # one epoch has, to exercise the epoch-rollover path.
    ds = NextTokenDataset(list(range(20)), context_length=8)  # len(ds) == 12
    it = iter_batches(ds, batch_size=4, seed=0)  # 3 batches/epoch
    batches = [next(it) for _ in range(10)]  # spans multiple epochs
    assert len(batches) == 10
    for x, y in batches:
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)


def test_iter_batches_skip_matches_unskipped_continuation() -> None:
    """The property Trainer.load_checkpoint relies on: skipping N batches
    reproduces exactly what consuming N batches from a fresh iterator and
    continuing would have yielded next."""
    ds = NextTokenDataset(list(range(300)), context_length=8)
    seed = 7
    skip = 9

    reference_iter = iter_batches(ds, batch_size=4, seed=seed)
    for _ in range(skip):
        next(reference_iter)
    expected_x, expected_y = next(reference_iter)

    skipped_iter = iter_batches(ds, batch_size=4, seed=seed, skip_batches=skip)
    actual_x, actual_y = next(skipped_iter)

    assert torch.equal(expected_x, actual_x)
    assert torch.equal(expected_y, actual_y)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_iter_batches_rejects_non_positive_batch_size(batch_size: int) -> None:
    ds = NextTokenDataset(list(range(50)), context_length=8)
    with pytest.raises(ValueError, match="batch_size"):
        next(iter_batches(ds, batch_size=batch_size, seed=0))


def test_iter_batches_rejects_negative_skip() -> None:
    ds = NextTokenDataset(list(range(50)), context_length=8)
    with pytest.raises(ValueError, match="skip_batches"):
        next(iter_batches(ds, batch_size=4, seed=0, skip_batches=-1))


def test_iter_batches_raises_instead_of_hanging_when_dataset_smaller_than_batch_size() -> None:
    """A dataset with fewer windows than batch_size would make every epoch
    of the internal drop_last=True DataLoader yield zero batches -- without
    an explicit guard, iter_batches's `while True` loop would spin forever
    without ever yielding a batch or raising, hanging any caller (e.g.
    Trainer.train_step/evaluate) instead of failing fast and clearly. This
    is reachable in practice with a small val split or a micro_batch_size
    misconfigured relative to a small corpus."""
    ds = NextTokenDataset(list(range(20)), context_length=8)  # len(ds) == 12
    with pytest.raises(ValueError, match="batch_size"):
        next(iter_batches(ds, batch_size=16, seed=0))


def test_iter_batches_accepts_dataset_exactly_batch_size() -> None:
    # Boundary case: len(dataset) == batch_size must still yield (exactly
    # one batch per epoch), not be treated as "too small".
    ds = NextTokenDataset(list(range(20)), context_length=8)  # len(ds) == 12
    x, y = next(iter_batches(ds, batch_size=12, seed=0))
    assert x.shape == (12, 8)
    assert y.shape == (12, 8)


def test_a_single_training_window_reconstructs_a_substring(
    tiny_corpus: str, trained_tokenizer: ByteLevelBPETokenizer
) -> None:
    # Decoding an *interior* window (not starting at position 0) is only
    # guaranteed to be valid UTF-8 for single-byte-per-character corpora --
    # tiny_corpus is pure ASCII, so this is safe here. See
    # docs/decisions/0002-tokenizer-scope.md's "known limitation" note.
    built = build_dataset(tiny_corpus, trained_tokenizer, context_length=6, val_fraction=0.2)
    full_train_text = trained_tokenizer.decode(built.train_dataset.tokens.tolist())

    idx = 3
    x, y = built.train_dataset[idx]
    window_tokens = built.train_dataset.tokens[idx : idx + 6 + 1].tolist()  # x plus final y token
    decoded_window = trained_tokenizer.decode(window_tokens)
    assert decoded_window in full_train_text
    assert torch.equal(x[1:], y[:-1])
