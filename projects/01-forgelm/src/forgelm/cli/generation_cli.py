"""Library-level generation operations the CLI wraps (also used by tests)."""

from __future__ import annotations

from typing import Any

from forgelm.config import GenerateConfig
from forgelm.generation import generate, load_model_from_checkpoint
from forgelm.tokenizer import ByteLevelBPETokenizer


def generate_text(config: GenerateConfig) -> dict[str, Any]:
    """Load a checkpoint + tokenizer and generate a continuation for
    ``config.prompt`` under ``config.generation`` (FR8, via a clean,
    checkpoint-driven entry point -- Phase 4 exit criterion)."""
    loaded = load_model_from_checkpoint(config.checkpoint_path, device=config.device)
    tokenizer = ByteLevelBPETokenizer.load(config.tokenizer_path)

    prompt_ids = tokenizer.encode(config.prompt, allow_special=False)
    if not prompt_ids:
        # An empty prompt (or one that encodes to nothing, e.g. "") still
        # needs at least one token to seed generation -- fall back to the
        # tokenizer's <|bos|> special token, the documented "start of
        # sequence" marker (forgelm.constants.DEFAULT_SPECIAL_TOKENS).
        bos = tokenizer.special_token_ids.get("<|bos|>")
        if bos is None:
            raise ValueError(
                "prompt encoded to zero tokens and the tokenizer has no '<|bos|>' "
                "special token to fall back to -- pass a non-empty prompt"
            )
        prompt_ids = [bos]

    generated_ids = generate(
        loaded.model,
        prompt_ids,
        config.generation,
        context_length=loaded.model_config.context_length,
        device=config.device,
    )
    generated_text = tokenizer.decode(generated_ids)

    return {
        "checkpoint_path": str(config.checkpoint_path),
        "step": loaded.step,
        "prompt": config.prompt,
        "generated_text": generated_text,
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated_ids) - len(prompt_ids),
    }
