"""Loss / perplexity evaluation of a model against a token dataset (FR9).

Component boundary: this module owns loss/perplexity computation only --
no training-loop machinery (optimizer, scheduler, checkpoint save/resume:
``forgelm.training``) and no benchmark timing (``forgelm.benchmarks``). It
is deliberately independent of ``forgelm.training.loop.Trainer`` for the
same reason ``forgelm.generation.checkpoint_loader`` is: evaluating a
saved checkpoint needs only a model + a dataset, not an optimizer.

Deterministic eval mode: batches are drawn from a plain, non-shuffled
``DataLoader`` (``shuffle=False``), so evaluation order needs no seed at
all and is trivially reproducible -- unlike ``Trainer.evaluate()``'s
seeded-but-still-randomly-ordered validation batches (adequate for a
periodic *during-training* signal, but a full deterministic walk over the
data is the more natural definition of "evaluate this checkpoint").
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Token-weighted average cross-entropy loss and derived perplexity
    over however many batches the configured token budget covered."""

    loss: float
    perplexity: float
    num_batches: int
    tokens_evaluated: int

    def to_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def evaluate_loss(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
) -> EvalResult:
    """Average cross-entropy loss + perplexity of ``model`` over
    ``dataset``, in eval mode with gradients disabled.

    Walks the dataset once in fixed (unshuffled) order. ``max_tokens``
    (FR9's "configurable token budget") caps how many target tokens are
    evaluated -- ``None`` evaluates the entire dataset. The loss is a
    token-weighted mean (``sum(per-token loss) / total tokens``, not a
    mean-of-batch-means), so a final partial batch does not get
    disproportionate weight. The budget is enforced exactly: if the batch
    that would cross ``max_tokens`` is reached, only the leading slice of
    its targets (up to the remaining budget) is counted, so
    ``tokens_evaluated`` never exceeds ``max_tokens``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive when set, got {max_tokens}")

    device = torch.device(device)
    was_training = model.training
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss_sum = 0.0
    total_tokens = 0
    num_batches = 0

    try:
        with torch.no_grad():
            for x, y in loader:
                if max_tokens is not None and total_tokens >= max_tokens:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                if max_tokens is not None:
                    remaining = max_tokens - total_tokens
                    if remaining < y_flat.numel():
                        # This batch would push total_tokens past max_tokens
                        # -- count only its first `remaining` target tokens
                        # so the reported (and loss-weighting) token count
                        # never overshoots the requested budget.
                        logits_flat = logits_flat[:remaining]
                        y_flat = y_flat[:remaining]
                batch_loss = F.cross_entropy(logits_flat, y_flat, reduction="sum")
                total_loss_sum += batch_loss.item()
                total_tokens += y_flat.numel()
                num_batches += 1
    finally:
        model.train(was_training)

    if num_batches == 0:
        raise ValueError(
            "evaluate_loss produced no batches -- dataset is empty or max_tokens is smaller "
            "than one example's worth of target tokens"
        )

    avg_loss = total_loss_sum / total_tokens
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        # An untrained/diverged checkpoint can have a loss large enough
        # that exp() overflows a Python float -- report +inf rather than
        # crashing the evaluator over a real (if extreme) result.
        perplexity = float("inf")
    return EvalResult(
        loss=avg_loss,
        perplexity=perplexity,
        num_batches=num_batches,
        tokens_evaluated=total_tokens,
    )
