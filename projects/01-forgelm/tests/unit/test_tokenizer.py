from __future__ import annotations

from pathlib import Path

import pytest

from forgelm.constants import BASE_BYTE_VOCAB_SIZE, DEFAULT_SPECIAL_TOKENS
from forgelm.tokenizer import ByteLevelBPETokenizer, compute_compression_stats
from forgelm.tokenizer.bpe import _pretokenize

# -- construction --------------------------------------------------------------


def test_construction_rejects_vocab_size_below_minimum() -> None:
    min_size = BASE_BYTE_VOCAB_SIZE + len(DEFAULT_SPECIAL_TOKENS)
    with pytest.raises(ValueError, match="vocab_size"):
        ByteLevelBPETokenizer(vocab_size=min_size - 1)


def test_untrained_tokenizer_raises_on_encode_decode() -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    with pytest.raises(RuntimeError):
        tok.encode("hi")
    with pytest.raises(RuntimeError):
        tok.decode([0])


# -- pre-tokenization -----------------------------------------------------------


def test_pretokenize_splits_into_word_punct_space_runs() -> None:
    assert _pretokenize("Hello, world!  42") == ["Hello", ",", " ", "world", "!", "  ", "42"]


def test_pretokenize_empty_string() -> None:
    assert _pretokenize("") == []


# -- round trip -------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [259, 270, 300, 400])
def test_round_trip_tiny_corpus(tiny_corpus: str, vocab_size: int) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=vocab_size)
    tok.train(tiny_corpus)
    ids = tok.encode(tiny_corpus)
    assert tok.decode(ids) == tiny_corpus


def test_round_trip_empty_string(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_round_trip_unicode_text(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    text = "café 你好 \U0001f600 naïve"  # accents, CJK, emoji
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_round_trip_text_unseen_during_training(tiny_corpus: str) -> None:
    # Byte-level BPE has zero OOV: any text can be encoded, even text with
    # characters never seen during training.
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    text = "zzz αβγ completely different text!! ###"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


# -- special tokens --------------------------------------------------------------


def test_special_tokens_round_trip(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    text = "<|bos|>hello there<|eos|>"
    ids = tok.encode(text)
    assert tok.special_token_ids["<|bos|>"] in ids
    assert tok.special_token_ids["<|eos|>"] in ids
    assert tok.decode(ids) == text


def test_special_tokens_have_fixed_ids_at_top_of_vocab_size() -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300, special_tokens=("<|bos|>", "<|eos|>", "<|pad|>"))
    tok.train("some short corpus for training merges here")
    assert tok.special_token_ids == {"<|bos|>": 297, "<|eos|>": 298, "<|pad|>": 299}


def test_allow_special_false_encodes_literal_text(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    text = "<|bos|>hi"
    ids = tok.encode(text, allow_special=False)
    assert tok.special_token_ids["<|bos|>"] not in ids
    assert tok.decode(ids) == text


# -- determinism ------------------------------------------------------------------


def test_training_is_deterministic(tiny_corpus: str) -> None:
    tok_a = ByteLevelBPETokenizer(vocab_size=320)
    tok_a.train(tiny_corpus)
    tok_b = ByteLevelBPETokenizer(vocab_size=320)
    tok_b.train(tiny_corpus)
    assert tok_a.merges == tok_b.merges
    assert tok_a.encode(tiny_corpus) == tok_b.encode(tiny_corpus)


def test_more_vocab_never_increases_token_count(tiny_corpus: str) -> None:
    small = ByteLevelBPETokenizer(vocab_size=270)
    small.train(tiny_corpus)
    big = ByteLevelBPETokenizer(vocab_size=400)
    big.train(tiny_corpus)
    assert len(big.encode(tiny_corpus)) <= len(small.encode(tiny_corpus))


# -- save / load --------------------------------------------------------------------


def test_save_load_round_trip(tiny_corpus: str, tmp_path: Path) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=320)
    tok.train(tiny_corpus)
    path = tmp_path / "nested" / "tokenizer.json"
    tok.save(path)
    assert path.exists()

    loaded = ByteLevelBPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    assert loaded.special_token_ids == tok.special_token_ids
    assert loaded.encode(tiny_corpus) == tok.encode(tiny_corpus)
    assert loaded.decode(loaded.encode(tiny_corpus)) == tiny_corpus


def test_save_before_train_raises(tmp_path: Path) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    with pytest.raises(RuntimeError):
        tok.save(tmp_path / "x.json")


# -- decode error handling -----------------------------------------------------------


def test_decode_unknown_id_raises(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    with pytest.raises(ValueError, match="unknown token id"):
        tok.decode([10_000])


# -- compression statistics --------------------------------------------------------


def test_compression_stats_basic_sanity(tiny_corpus: str) -> None:
    tok = ByteLevelBPETokenizer(vocab_size=320)
    tok.train(tiny_corpus)
    stats = compute_compression_stats(tok, tiny_corpus)
    assert stats["num_bytes"] == float(len(tiny_corpus.encode("utf-8")))
    assert stats["num_chars"] == float(len(tiny_corpus))
    assert stats["num_tokens"] > 0
    assert stats["bytes_per_token"] >= 1.0  # BPE tokens are never sub-byte


def test_compression_improves_with_more_merges(tiny_corpus: str) -> None:
    small = ByteLevelBPETokenizer(vocab_size=270)
    small.train(tiny_corpus)
    big = ByteLevelBPETokenizer(vocab_size=400)
    big.train(tiny_corpus)
    small_stats = compute_compression_stats(small, tiny_corpus)
    big_stats = compute_compression_stats(big, tiny_corpus)
    assert big_stats["bytes_per_token"] >= small_stats["bytes_per_token"]
