"""Phase 2 exit criterion: "forward pass numerically sane, shape-safe, and
trainable on toy overfit dataset."

Deliberately self-contained (raw torch training loop, plain
``torch.optim.AdamW``) rather than going through ``forgelm.training`` --
Phase 2 is about the *model*'s trainability in isolation, independent of
Phase 3's training-system machinery, which has its own overfit-adjacent
coverage in ``tests/integration/test_checkpoint_resume.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forgelm.config import ModelConfig
from forgelm.model import TransformerDecoder
from forgelm.seed import set_seed


def test_tiny_model_overfits_a_toy_repeating_sequence() -> None:
    set_seed(0)
    vocab_size = 12
    context_length = 8
    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    model = TransformerDecoder(config)

    # A short, easily-memorizable repeating pattern -- next-token
    # prediction on this is trivially learnable (predict "the next number
    # in the cycle") if the forward pass and gradients are wired up
    # correctly end to end.
    pattern = list(range(vocab_size))
    tokens = torch.tensor((pattern * 20)[: context_length + 1], dtype=torch.long)
    x = tokens[:-1].unsqueeze(0)
    y = tokens[1:].unsqueeze(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.train()
    losses: list[float] = []
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        assert torch.isfinite(logits).all(), "forward pass produced non-finite logits"
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]

    # Cross-entropy over `vocab_size` uniform-ish classes starts near
    # log(vocab_size); after 200 steps of overfitting eight positions of a
    # near-memorizable sequence, loss should have dropped by more than an
    # order of magnitude and be close to zero.
    assert final_loss < 0.05
    assert final_loss < initial_loss * 0.1

    # Loss should have generally decreased, not just at the final step --
    # guard against a test that only "passes" due to one lucky late step.
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg


def test_model_forward_pass_is_shape_safe_across_batch_and_seq_len() -> None:
    """A lighter shape-safety check spanning several (batch, seq_len)
    combinations, independent of the overfit-training test above."""
    config = ModelConfig(
        vocab_size=32, context_length=16, d_model=16, n_layers=2, n_heads=4, d_ff=32
    )
    model = TransformerDecoder(config)
    model.eval()

    for batch, seq_len in [(1, 1), (4, 5), (2, 16)]:
        token_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        with torch.no_grad():
            logits = model(token_ids)
        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()
