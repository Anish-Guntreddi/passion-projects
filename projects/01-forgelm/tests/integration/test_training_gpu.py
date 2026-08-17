"""GPU smoke test for the Phase 2/3 model + training system, run when
hardware/CI permits (spec §1.8). Automatically skipped (not failed) when
no CUDA device is visible, mirroring
``tests/integration/test_smoke_gpu.py``'s pattern from Phase 0.
"""

from __future__ import annotations

import pytest
import torch

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.dataset import NextTokenDataset
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed
from forgelm.training import Trainer
from forgelm.training.precision import resolve_precision

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=32, context_length=8, d_model=16, n_layers=2, n_heads=2, d_ff=32)


@requires_cuda
def test_decoder_forward_pass_runs_on_cuda() -> None:
    set_seed(0)
    config = _tiny_model_config()
    model = TransformerDecoder(config).to("cuda")
    token_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    logits = model(token_ids)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()


@requires_cuda
def test_auto_precision_resolves_to_bf16_on_this_gpu_if_supported() -> None:
    policy = resolve_precision(torch.device("cuda"), "auto")
    assert policy.device_type == "cuda"
    if torch.cuda.is_bf16_supported():
        assert policy.autocast_dtype is torch.bfloat16
    else:
        assert policy.autocast_dtype is None


@requires_cuda
def test_trainer_runs_a_few_steps_on_cuda() -> None:
    set_seed(0)
    model_config = _tiny_model_config()
    training_config = TrainingConfig(
        micro_batch_size=4,
        max_steps=5,
        warmup_steps=1,
        eval_interval=1000,
        device="cuda",
        precision="auto",
    )
    tokens = [i % model_config.vocab_size for i in range(200)]
    train_ds = NextTokenDataset(tokens, model_config.context_length)
    val_ds = NextTokenDataset(tokens, model_config.context_length)

    model = TransformerDecoder(model_config)
    trainer = Trainer(model, train_ds, val_ds, model_config, training_config)

    losses = [trainer.train_step().loss for _ in range(5)]
    assert all(loss == loss for loss in losses)  # no NaNs
    assert all(loss >= 0.0 for loss in losses)

    val_loss = trainer.evaluate()
    assert val_loss == val_loss and val_loss >= 0.0
