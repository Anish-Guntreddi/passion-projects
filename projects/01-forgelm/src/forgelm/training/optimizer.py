"""AdamW optimizer construction with decay / no-decay parameter groups.

Weight decay is applied only to >=2-D parameters (this architecture's
``Linear`` weight matrices and the tied embedding); 1-D parameters
(RMSNorm gains -- this architecture has no biases anywhere, see
``forgelm.model``) are excluded, following the practice popularized by
GPT-2/nanoGPT-style implementations: decaying a normalization gain
toward zero has no principled regularization benefit and empirically
hurts small-model training.

``torch.optim.AdamW`` itself is used directly (FR6: "AdamW" is a stack
choice, not something this project reimplements) -- what ForgeLM
contributes here is the parameter grouping and how it wires into the
rest of the training config, not the optimizer math itself.
"""

from __future__ import annotations

import torch
from torch import nn

from forgelm.config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Build an ``AdamW`` optimizer with decay/no-decay parameter groups."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.lr, betas=(config.beta1, config.beta2))
