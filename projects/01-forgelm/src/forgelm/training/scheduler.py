"""Learning-rate schedule: linear warmup, then cosine decay to
``min_lr_ratio * lr`` by ``max_steps`` -- a standard LLM-pretraining
schedule (e.g. GPT-3, LLaMA).

:func:`lr_multiplier` is a plain, pure function of
``(step, warmup_steps, max_steps, min_lr_ratio)`` -- the schedule *shape*
has no dependency on ``torch`` optimizer/scheduler state at all, so it is
directly unit-testable on its own. :func:`build_scheduler` is the only
place that wires it into ``torch.optim.lr_scheduler.LambdaLR``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from forgelm.config import TrainingConfig


def lr_multiplier(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    """Fraction of the base LR to use at ``step`` (0-indexed scheduler step).

    - ``step < warmup_steps``: linear ramp from ``1/warmup_steps`` to ``1.0``.
    - ``warmup_steps <= step < max_steps``: cosine decay from ``1.0`` down
      to ``min_lr_ratio``.
    - ``step >= max_steps``: held at ``min_lr_ratio``.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def make_lr_lambda(config: TrainingConfig) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        return lr_multiplier(step, config.warmup_steps, config.max_steps, config.min_lr_ratio)

    return lr_lambda


def build_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    """Build a ``LambdaLR`` that applies :func:`lr_multiplier` per step."""
    return LambdaLR(optimizer, lr_lambda=make_lr_lambda(config))
