from __future__ import annotations

import pytest
import torch
from torch import nn

from forgelm.config import TrainingConfig
from forgelm.training.scheduler import build_scheduler, lr_multiplier

# -- lr_multiplier: pure-function schedule shape ----------------------------------


def test_lr_multiplier_linear_warmup() -> None:
    warmup, max_steps, min_ratio = 4, 20, 0.1
    values = [lr_multiplier(s, warmup, max_steps, min_ratio) for s in range(warmup)]
    assert values == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_lr_multiplier_peaks_at_one_right_after_warmup() -> None:
    warmup, max_steps, min_ratio = 5, 20, 0.1
    assert lr_multiplier(warmup, warmup, max_steps, min_ratio) == pytest.approx(1.0)


def test_lr_multiplier_reaches_min_ratio_at_max_steps() -> None:
    warmup, max_steps, min_ratio = 5, 20, 0.1
    assert lr_multiplier(max_steps, warmup, max_steps, min_ratio) == pytest.approx(min_ratio)


def test_lr_multiplier_held_at_min_ratio_past_max_steps() -> None:
    warmup, max_steps, min_ratio = 5, 20, 0.1
    assert lr_multiplier(max_steps + 50, warmup, max_steps, min_ratio) == pytest.approx(min_ratio)


def test_lr_multiplier_monotonically_decreases_during_cosine_decay() -> None:
    warmup, max_steps, min_ratio = 5, 30, 0.1
    values = [lr_multiplier(s, warmup, max_steps, min_ratio) for s in range(warmup, max_steps + 1)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_lr_multiplier_never_below_min_ratio_or_above_one() -> None:
    warmup, max_steps, min_ratio = 3, 25, 0.2
    for step in range(0, max_steps + 10):
        value = lr_multiplier(step, warmup, max_steps, min_ratio)
        assert min_ratio - 1e-9 <= value <= 1.0 + 1e-9


def test_lr_multiplier_zero_warmup_starts_at_peak() -> None:
    assert lr_multiplier(0, 0, 10, 0.1) == pytest.approx(1.0)


def test_lr_multiplier_rejects_negative_step() -> None:
    with pytest.raises(ValueError):
        lr_multiplier(-1, 5, 20, 0.1)


# -- build_scheduler: wiring into torch's LambdaLR ---------------------------------


def test_build_scheduler_lr_trace_matches_lr_multiplier() -> None:
    config = TrainingConfig(lr=0.1, warmup_steps=3, max_steps=10, min_lr_ratio=0.1)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = build_scheduler(optimizer, config)

    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(12):
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    expected = [
        config.lr * lr_multiplier(step, config.warmup_steps, config.max_steps, config.min_lr_ratio)
        for step in range(len(observed))
    ]
    for obs, exp in zip(observed, expected, strict=True):
        assert obs == pytest.approx(exp, rel=1e-6, abs=1e-9)
