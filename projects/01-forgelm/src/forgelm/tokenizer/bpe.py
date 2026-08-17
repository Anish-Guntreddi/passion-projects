"""A minimal, from-scratch byte-level BPE tokenizer.

Design (see ``docs/decisions/0002-tokenizer-scope.md`` for the D1 sign-off):

- The base vocabulary is the 256 raw byte values (ids 0-255). Because every
  Unicode string is a valid UTF-8 byte sequence, this tokenizer can encode
  *any* input text with zero out-of-vocabulary tokens -- there is no
  ``<unk>``.
- Training learns a fixed number of merge rules over byte pairs, exactly
  like the algorithm in Sennrich et al. (2016) "Neural Machine Translation
  of Rare Words with Subword Units": repeatedly merge the most frequent
  adjacent symbol pair into a new symbol. Ties are broken deterministically
  by pair value so training is 100% reproducible for a given corpus.
- Text is pre-tokenized into runs of letters / digits / other-punctuation /
  whitespace before BPE is applied, so merges never span a word boundary
  (this keeps the induced vocabulary sane and mirrors the class of
  pre-tokenization regexes used by most modern BPE tokenizers, without
  reproducing any specific library's pattern or code).
- A handful of special tokens (``<|bos|>``, ``<|eos|>``, ``<|pad|>`` by
  default) are reserved at the *end* of the id space and never produced by
  the byte-merge machinery; :meth:`encode` recognizes their literal string
  form in input text and emits the reserved id directly.

This file intentionally implements everything from first principles: no
tiktoken, sentencepiece, or HuggingFace ``tokenizers`` calls anywhere below.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from forgelm.constants import BASE_BYTE_VOCAB_SIZE, DEFAULT_SPECIAL_TOKENS

# Runs of: ASCII letters | digits | "other" (punctuation/symbols) | whitespace.
# Splitting on these boundaries keeps merges from crossing between, e.g., a
# word and the punctuation after it, without hand-coding any specific
# tokenizer's exact regex.
_PRETOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]+|\s+")

Pair = tuple[int, int]
Merge = tuple[Pair, int]


def _pretokenize(text: str) -> list[str]:
    """Split text into words/punctuation-runs/whitespace-runs."""
    return _PRETOKEN_RE.findall(text)


def _merge_all(
    word_freqs: Counter[tuple[int, ...]], pair: Pair, new_id: int
) -> Counter[tuple[int, ...]]:
    """Replace every adjacent occurrence of ``pair`` with ``new_id``."""
    a, b = pair
    merged: Counter[tuple[int, ...]] = Counter()
    for word, freq in word_freqs.items():
        out: list[int] = []
        i, n = 0, len(word)
        while i < n:
            if i < n - 1 and word[i] == a and word[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        merged[tuple(out)] += freq
    return merged


class ByteLevelBPETokenizer:
    """A trainable, deterministic byte-level BPE tokenizer.

    Attributes:
        vocab_size: the *configured target* vocabulary size (256 base bytes
            + learned merges + special tokens). If the training corpus is
            too small/repetitive to yield that many distinct merges,
            ``num_merges`` will be smaller than requested and some ids
            between the last merge and the first special token are simply
            unused -- this is safe (never emitted, never required by
            decode) and only affects tiny toy corpora.
        special_tokens: reserved token strings, in id order (highest ids).
    """

    def __init__(
        self,
        vocab_size: int = 512,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        special_tokens = tuple(special_tokens)
        min_size = BASE_BYTE_VOCAB_SIZE + len(special_tokens)
        if vocab_size < min_size:
            raise ValueError(
                f"vocab_size must be >= {min_size} (256 base bytes + "
                f"{len(special_tokens)} special tokens), got {vocab_size}"
            )
        self.vocab_size = vocab_size
        self.special_tokens: tuple[str, ...] = special_tokens
        self.merges: list[Merge] = []

        self._vocab: dict[int, bytes] | None = None
        self._merge_rank: dict[Pair, int] = {}
        self._merge_pair_to_id: dict[Pair, int] = {}
        self.special_token_ids: dict[str, int] = {}
        self.special_id_to_token: dict[int, str] = {}

    # -- state ------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._vocab is not None

    @property
    def num_merges(self) -> int:
        return len(self.merges)

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("tokenizer has no vocabulary yet; call train() or load() first")

    def _rebuild_indices(self) -> None:
        """Recompute derived lookup tables from ``self.merges``. Idempotent."""
        self._merge_rank = {pair: rank for rank, (pair, _new_id) in enumerate(self.merges)}
        self._merge_pair_to_id = {pair: new_id for pair, new_id in self.merges}

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BASE_BYTE_VOCAB_SIZE)}
        for pair, new_id in self.merges:
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        self._vocab = vocab

        specials_start = self.vocab_size - len(self.special_tokens)
        self.special_token_ids = {
            tok: specials_start + i for i, tok in enumerate(self.special_tokens)
        }
        self.special_id_to_token = {v: k for k, v in self.special_token_ids.items()}

    # -- training -----------------------------------------------------------

    def train(self, text: str) -> None:
        """Learn merge rules from ``text``. Deterministic given the same text."""
        word_freqs: Counter[tuple[int, ...]] = Counter()
        for word in _pretokenize(text):
            word_freqs[tuple(word.encode("utf-8"))] += 1

        target_id = self.vocab_size - len(self.special_tokens)
        merges: list[Merge] = []
        next_id = BASE_BYTE_VOCAB_SIZE
        while next_id < target_id:
            pair_counts: Counter[Pair] = Counter()
            for word, freq in word_freqs.items():
                for a, b in zip(word, word[1:], strict=False):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break  # corpus exhausted: no more adjacent pairs to merge
            # Deterministic tie-break: highest count, then highest pair value.
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merges.append((best_pair, next_id))
            word_freqs = _merge_all(word_freqs, best_pair, next_id)
            next_id += 1

        self.merges = merges
        self._rebuild_indices()

    # -- encode / decode ------------------------------------------------------

    def _encode_word(self, word: bytes) -> list[int]:
        """Apply learned merges to one pre-token, in learned priority order."""
        ids: list[int] = list(word)
        while len(ids) > 1:
            best_rank: int | None = None
            best_pos = -1
            for i in range(len(ids) - 1):
                rank = self._merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_pos == -1:
                break
            pair = (ids[best_pos], ids[best_pos + 1])
            new_id = self._merge_pair_to_id[pair]
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2 :]
        return ids

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """Encode text to token ids. Lossless: ``decode(encode(x)) == x``."""
        self._require_trained()
        if not text:
            return []

        if allow_special and self.special_tokens:
            pattern = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_token_ids:
                ids.append(self.special_token_ids[part])
                continue
            for word in _pretokenize(part):
                ids.extend(self._encode_word(word.encode("utf-8")))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text.

        Guaranteed exact (``decode(encode(x)) == x``) for a *complete*
        id sequence produced by :meth:`encode`. This is a general property
        of byte-level BPE worth being explicit about: merges group raw
        *bytes*, without regard for Unicode character boundaries, so a
        learned token can represent a byte span that starts or ends mid
        multi-byte UTF-8 character. Decoding a slice of ids that does not
        start at position 0 or end at the true end of an encoded sequence
        (e.g. an arbitrary interior window) is therefore only guaranteed
        to succeed for single-byte-per-character (e.g. ASCII-only) text;
        for general Unicode text, decode the full id sequence a split
        produced (e.g. ``decode(train_ids + val_ids)``, not
        ``decode(train_ids) + decode(val_ids)``) when exactness matters.
        See ``docs/decisions/0002-tokenizer-scope.md``.
        """
        self._require_trained()
        assert self._vocab is not None
        pieces: list[str] = []
        buf = bytearray()
        for token_id in ids:
            if token_id in self.special_id_to_token:
                if buf:
                    pieces.append(bytes(buf).decode("utf-8"))
                    buf = bytearray()
                pieces.append(self.special_id_to_token[token_id])
                continue
            piece = self._vocab.get(token_id)
            if piece is None:
                raise ValueError(
                    f"unknown token id {token_id}; trained vocab covers "
                    f"0..{BASE_BYTE_VOCAB_SIZE + self.num_merges - 1} plus special ids "
                    f"{sorted(self.special_id_to_token)}"
                )
            buf.extend(piece)
        if buf:
            pieces.append(bytes(buf).decode("utf-8"))
        return "".join(pieces)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save merges + config as JSON. The learned vocab is derivable from
        merges alone, so only the compact merge list is persisted."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> ByteLevelBPETokenizer:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        tok = cls(
            vocab_size=payload["vocab_size"],
            special_tokens=tuple(payload["special_tokens"]),
        )
        tok.merges = [((a, b), new_id) for a, b, new_id in payload["merges"]]
        tok._rebuild_indices()
        return tok


def compute_compression_stats(tokenizer: ByteLevelBPETokenizer, text: str) -> dict[str, float]:
    """Compression statistics for ``text`` under ``tokenizer`` (FR2)."""
    ids = tokenizer.encode(text, allow_special=False)
    num_bytes = len(text.encode("utf-8"))
    num_chars = len(text)
    num_tokens = len(ids)
    return {
        "num_chars": float(num_chars),
        "num_bytes": float(num_bytes),
        "num_tokens": float(num_tokens),
        "bytes_per_token": (num_bytes / num_tokens) if num_tokens else 0.0,
        "chars_per_token": (num_chars / num_tokens) if num_tokens else 0.0,
    }
