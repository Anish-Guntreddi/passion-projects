"""Text -> token-array -> fixed-length next-token example pipeline (FR3).

Component boundary: this module owns *datasets, packing, and splits*. It
knows nothing about model architecture and nothing about training loops --
it hands back plain tensors.

Pipeline:

    raw text file
        -> load_text()
        -> tokenizer.encode()            (forgelm.tokenizer)
        -> train_val_split()             deterministic, contiguous
        -> NextTokenDataset(tokens, L)   fixed-length (x, y) pairs
        -> make_dataloader()             deterministic batch order given a seed

Because the tokenizer is lossless byte-level BPE, any contiguous slice of
token ids decodes back to the exact substring of the original text that
produced it -- this is what makes training batches "reconstructable" (the
Phase 1 exit criterion).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from forgelm.tokenizer import ByteLevelBPETokenizer


def load_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a local text file. No network, no implicit downloading (D3)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset text file not found: {path}")
    return path.read_text(encoding=encoding)


def tokenize_corpus(text: str, tokenizer: ByteLevelBPETokenizer) -> list[int]:
    """Encode an entire corpus to a flat list of token ids (no specials)."""
    return tokenizer.encode(text, allow_special=False)


def train_val_split(
    tokens: Sequence[int], val_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Deterministic contiguous split of a token stream.

    A contiguous split (rather than a random per-token shuffle) is used
    deliberately: the objective is next-token prediction over a stream, so
    shuffling individual tokens would destroy the local context each
    training example depends on. The split point is a pure function of
    ``len(tokens)`` and ``val_fraction`` -- no RNG involved, so it is
    trivially reproducible without needing to record a seed.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n = len(tokens)
    split_idx = int(n * (1.0 - val_fraction))
    train_tokens = list(tokens[:split_idx])
    val_tokens = list(tokens[split_idx:])
    return train_tokens, val_tokens


class NextTokenDataset(Dataset):
    """Fixed-length next-token windows over a flat token array (FR3).

    ``__getitem__(i)`` returns ``(x, y)`` with ``x = tokens[i : i+L]`` and
    ``y = tokens[i+1 : i+L+1]`` -- the standard shifted-by-one next-token
    objective input. Windows overlap (stride 1), which is standard for
    small-corpus LM training and keeps every token position covered as both
    a context token and a prediction target.

    Windows are plain integer id tensors -- this class never decodes them,
    and nothing downstream should call ``tokenizer.decode()`` on an
    arbitrary window in isolation for non-ASCII corpora: byte-level BPE
    merges group raw bytes, not Unicode characters, so an interior window's
    start/end need not fall on a UTF-8 character boundary. This is a known,
    accepted property of byte-level BPE (see
    ``docs/decisions/0002-tokenizer-scope.md``'s "known limitation" and
    :meth:`ByteLevelBPETokenizer.decode`'s docstring), not a defect here --
    the Phase 1 "reconstructable" exit criterion is about reconstructing the
    full corpus from its full id sequence (``decode(train_ids + val_ids) ==
    original_text``, verified on the real non-ASCII example corpus by
    ``tests/integration/test_pipeline.py``), not about arbitrary interior
    slices decoding standalone.
    """

    def __init__(self, tokens: Sequence[int], context_length: int) -> None:
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if len(tokens) < context_length + 1:
            raise ValueError(
                f"need at least context_length + 1 = {context_length + 1} tokens, got {len(tokens)}"
            )
        self.tokens = torch.tensor(list(tokens), dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """A DataLoader whose batch order is a deterministic function of ``seed``.

    ``num_workers=0`` by default: single-process iteration keeps batch order
    fully determined by the ``torch.Generator`` seed. Multi-worker loading
    is only safe to enable once a ``worker_init_fn`` (see
    ``forgelm.seed.seed_worker``) is wired up by the caller.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def iter_batches(
    dataset: Dataset, batch_size: int, seed: int, *, skip_batches: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite, deterministic stream of ``(x, y)`` batches over ``dataset``.

    Unlike :func:`make_dataloader` (a single-epoch DataLoader whose
    permutation is fixed only until the next ``iter()`` call advances its
    generator's internal state), this is a *pure* function of ``(dataset,
    batch_size, seed)``: epoch ``e``'s batches always come from
    ``make_dataloader(dataset, batch_size, seed=seed + e)``, constructed
    fresh each time -- so the entire infinite batch sequence produced from
    step 0 onward is reproducible from those three arguments alone, with
    no hidden state carried between calls.

    This is what makes training resumable (Phase 3 exit criterion): to
    continue a run from training step ``N``, recreate this generator with
    the same ``(dataset, batch_size, seed)`` and pass ``skip_batches=N`` to
    fast-forward past the batches already consumed before the checkpoint
    was saved, reproducing the exact batch order the uninterrupted run
    would have seen at steps ``N, N+1, ...``. See
    ``docs/decisions/0009-checkpoint-format.md`` and
    ``forgelm.training.loop.Trainer``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if skip_batches < 0:
        raise ValueError(f"skip_batches must be non-negative, got {skip_batches}")
    try:
        dataset_len = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "iter_batches requires a Dataset that implements __len__ (needed to "
            "guard against a batch_size larger than the dataset -- see below)"
        ) from exc
    if dataset_len < batch_size:
        # make_dataloader() is always called with drop_last=True below, so
        # an epoch over a dataset smaller than batch_size yields *zero*
        # batches -- without this check the `while True` loop below would
        # spin forever without ever yielding a batch or raising, hanging
        # any caller (e.g. Trainer.train_step()/evaluate()) instead of
        # failing with a clear, immediate error.
        raise ValueError(
            f"dataset has {dataset_len} example(s), fewer than batch_size={batch_size}; "
            "with drop_last=True this would never yield a full batch. Reduce "
            "micro_batch_size, increase context_length's window count, or use a "
            "larger dataset/val split."
        )

    batches_yielded = 0
    epoch = 0
    while True:
        loader = make_dataloader(
            dataset, batch_size, shuffle=True, seed=seed + epoch, drop_last=True
        )
        for x, y in loader:
            if batches_yielded >= skip_batches:
                yield x, y
            batches_yielded += 1
        epoch += 1


@dataclasses.dataclass(frozen=True)
class DatasetStats:
    """Dataset statistics (FR2/FR3)."""

    num_chars: int
    num_bytes: int
    num_tokens: int
    unique_tokens: int
    bytes_per_token: float
    train_tokens: int
    val_tokens: int
    val_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def compute_dataset_stats(text: str, tokens: Sequence[int], val_fraction: float) -> DatasetStats:
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokens)
    train_tokens, val_tokens = train_val_split(tokens, val_fraction)
    return DatasetStats(
        num_chars=len(text),
        num_bytes=num_bytes,
        num_tokens=num_tokens,
        unique_tokens=len(set(tokens)),
        bytes_per_token=(num_bytes / num_tokens) if num_tokens else 0.0,
        train_tokens=len(train_tokens),
        val_tokens=len(val_tokens),
        val_fraction=val_fraction,
    )


@dataclasses.dataclass
class BuiltDataset:
    """Everything :func:`build_dataset` produces, ready for training/tests."""

    train_dataset: NextTokenDataset
    val_dataset: NextTokenDataset
    stats: DatasetStats


def build_dataset(
    text: str,
    tokenizer: ByteLevelBPETokenizer,
    context_length: int,
    val_fraction: float = 0.1,
) -> BuiltDataset:
    """End-to-end: text -> tokens -> split -> two NextTokenDatasets + stats."""
    tokens = tokenize_corpus(text, tokenizer)
    stats = compute_dataset_stats(text, tokens, val_fraction)
    train_tokens, val_tokens = train_val_split(tokens, val_fraction)
    return BuiltDataset(
        train_dataset=NextTokenDataset(train_tokens, context_length),
        val_dataset=NextTokenDataset(val_tokens, context_length),
        stats=stats,
    )
