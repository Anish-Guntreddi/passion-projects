"""Autoregressive text generation from a trained :class:`TransformerDecoder`.

Component boundary: this module owns *sampling*, nothing else -- it takes
an already-constructed, already-loaded model (see
``forgelm.generation.checkpoint_loader``) and a prompt's token ids, and
returns a longer token-id sequence. It has no notion of checkpoints,
tokenizers, or CLI plumbing.

Decoding modes (FR8):

- **Greedy** (``config.do_sample=False``): argmax of the logits at every
  step. Fully deterministic -- the same prompt always produces the same
  continuation, regardless of ``seed``.
- **Sampling** (``config.do_sample=True``): logits are divided by
  ``temperature`` (lower = sharper/more deterministic, higher = flatter/
  more random), then optionally restricted to the top ``top_k`` logits
  and/or the smallest nucleus of tokens whose cumulative probability
  reaches ``top_p`` (Holtzman et al. 2019, "The Curious Case of Neural
  Text Degeneration"), then sampled from with a seeded
  ``torch.Generator`` -- so a run is reproducible from
  ``(prompt, GenerationSettings)`` alone, independent of any other RNG
  consumption elsewhere in the process (e.g. training).

No KV cache: each step re-runs the full forward pass over the current
(context-length-cropped) sequence. This is the correctness-first choice
appropriate for this project's toy/portfolio scale (spec §1.7 "implement
correctness before optimization") -- a KV cache is a pure performance
optimization with no effect on *what* gets generated, and is intentionally
left as a documented future optimization rather than added speculative
complexity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from forgelm.config import GenerationSettings

_NEG_INF = float("-inf")


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` highest logits per row; set the rest to -inf.

    ``logits``: shape ``(batch, vocab_size)``.

    Builds the keep-mask from ``topk``'s own *indices* rather than
    thresholding on its k-th value: comparing every logit against a
    ``>= threshold`` cutoff would keep every tie at the boundary too (e.g.
    ``top_k=2`` over ``[5, 4, 4, 1]`` has two values tied for 2nd place, so
    a threshold-based filter would keep 3 candidates, silently violating
    the "exactly top_k candidates" constraint FR8 asks for). Scattering the
    top-k indices back onto a ``-inf``-filled tensor keeps exactly ``k``
    entries per row regardless of ties elsewhere in the distribution.
    """
    k = min(top_k, logits.size(-1))
    topk = torch.topk(logits, k, dim=-1)
    filtered = torch.full_like(logits, _NEG_INF)
    filtered.scatter_(-1, topk.indices, topk.values)
    return filtered


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest prefix of (probability-sorted,
    descending) tokens whose cumulative probability is >= ``top_p``; set
    every other logit to -inf.

    ``logits``: shape ``(batch, vocab_size)``.
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Token i is dropped once the cumulative probability *before* it already
    # reached top_p -- i.e. shift the "exceeds top_p" mask right by one so
    # the token that first crosses the threshold is always kept (without
    # this shift, a top_p smaller than the single most likely token's own
    # probability would drop every token and leave nothing to sample).
    sorted_remove = cumulative > top_p
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False

    remove = torch.zeros_like(sorted_remove).scatter(1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove, _NEG_INF)


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationSettings,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id for each row of ``logits`` (shape ``(batch,
    vocab_size)``, the *last position's* logits only -- callers slice
    ``model(x)[:, -1, :]`` before calling this).

    Returns a ``(batch, 1)`` LongTensor. Greedy when
    ``config.do_sample=False`` (``temperature``/``top_k``/``top_p`` are
    then ignored); otherwise samples from the filtered, temperature-scaled
    softmax distribution using ``generator``.
    """
    if not config.do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / config.temperature
    if config.top_k is not None:
        scaled = _top_k_filter(scaled, config.top_k)
    if config.top_p is not None:
        scaled = _top_p_filter(scaled, config.top_p)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: list[int],
    config: GenerationSettings,
    *,
    context_length: int,
    device: str | torch.device = "cpu",
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate up to ``config.max_new_tokens`` tokens continuing
    ``prompt_ids``. Returns the full sequence (prompt + generated tokens);
    generation stops early if ``eos_token_id`` is produced.

    ``model`` is run in eval mode for the duration of this call and
    restored to its prior ``.training`` flag afterward, so calling this
    mid-training-loop (e.g. periodic qualitative samples) would not
    silently leave dropout disabled for the next training step.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    device = torch.device(device)
    was_training = model.training
    model.eval()
    generator = None
    if config.do_sample:
        # A dedicated generator (not the global RNG stream) so repeated
        # calls with the same GenerationSettings are reproducible
        # independent of how much RNG state training/other calls already
        # consumed. Must live on the same device as the sampled tensor --
        # torch.multinomial requires a matching-device generator.
        generator = torch.Generator(device=device.type)
        generator.manual_seed(config.seed)

    tokens = list(prompt_ids)
    try:
        for _ in range(config.max_new_tokens):
            window = tokens[-context_length:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]  # (1, vocab_size) -- last position only
            next_id = sample_next_token(logits, config, generator).item()
            tokens.append(int(next_id))
            if eos_token_id is not None and next_id == eos_token_id:
                break
    finally:
        model.train(was_training)

    return tokens
