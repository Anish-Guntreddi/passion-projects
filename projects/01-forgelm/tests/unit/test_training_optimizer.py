from __future__ import annotations

import torch
from torch import nn

from forgelm.config import TrainingConfig
from forgelm.model import RMSNorm
from forgelm.training import build_optimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=True)
        self.norm = RMSNorm(4)


def test_build_optimizer_is_adamw_with_configured_lr_and_betas() -> None:
    model = _TinyModel()
    config = TrainingConfig(lr=1e-3, beta1=0.8, beta2=0.9, weight_decay=0.1)
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-3
        assert group["betas"] == (0.8, 0.9)


def test_build_optimizer_separates_decay_and_no_decay_by_ndim() -> None:
    model = _TinyModel()
    config = TrainingConfig(weight_decay=0.25)
    optimizer = build_optimizer(model, config)

    assert len(optimizer.param_groups) == 2
    decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.25)
    no_decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.0)

    decay_shapes = {tuple(p.shape) for p in decay_group["params"]}
    no_decay_shapes = {tuple(p.shape) for p in no_decay_group["params"]}

    # linear.weight is 2-D -> decay; linear.bias and norm.weight are 1-D -> no decay.
    assert tuple(model.linear.weight.shape) in decay_shapes
    assert tuple(model.linear.bias.shape) in no_decay_shapes
    assert tuple(model.norm.weight.shape) in no_decay_shapes
    assert all(len(shape) >= 2 for shape in decay_shapes)
    assert all(len(shape) < 2 for shape in no_decay_shapes)


def test_build_optimizer_covers_every_trainable_parameter_exactly_once() -> None:
    model = _TinyModel()
    optimizer = build_optimizer(model, TrainingConfig())
    optimized_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    model_param_ids = [id(p) for p in model.parameters()]
    assert sorted(optimized_ids) == sorted(model_param_ids)
    assert len(optimized_ids) == len(set(optimized_ids))  # no parameter counted twice
