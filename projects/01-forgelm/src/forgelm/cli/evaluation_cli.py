"""Library-level evaluation / sample-report operations the CLI wraps (also
used by tests)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from forgelm.config import EvaluateConfig, SampleReportConfig
from forgelm.dataset import NextTokenDataset
from forgelm.evaluation import GenerationSample, SampleReport, evaluate_loss, save_report
from forgelm.generation import generate, load_model_from_checkpoint
from forgelm.tokenizer import ByteLevelBPETokenizer


def _load_token_dataset(path: str | Path, context_length: int) -> NextTokenDataset:
    tokens = np.load(path)
    return NextTokenDataset(tokens.tolist(), context_length)


def evaluate_checkpoint(config: EvaluateConfig) -> dict[str, Any]:
    """Loss/perplexity of a checkpoint against a pre-built token array
    (FR9, Phase 4 exit criterion: "trained checkpoint evaluates ... via
    clean CLI")."""
    loaded = load_model_from_checkpoint(config.checkpoint_path, device=config.device)
    dataset = _load_token_dataset(config.tokens_path, loaded.model_config.context_length)
    result = evaluate_loss(
        loaded.model,
        dataset,
        batch_size=config.batch_size,
        device=config.device,
        max_tokens=config.max_tokens,
    )
    return {
        "checkpoint_path": str(config.checkpoint_path),
        "step": loaded.step,
        **result.to_dict(),
    }


def build_sample_report(config: SampleReportConfig) -> dict[str, Any]:
    """Evaluate (optionally) + generate for every prompt in
    ``config.prompts``, and write the combined JSON + Markdown sample
    report (FR9's "sample-report format" deliverable)."""
    loaded = load_model_from_checkpoint(config.checkpoint_path, device=config.device)
    tokenizer = ByteLevelBPETokenizer.load(config.tokenizer_path)

    eval_result_dict: dict[str, Any] | None = None
    if config.val_tokens_path is not None:
        dataset = _load_token_dataset(config.val_tokens_path, loaded.model_config.context_length)
        eval_result = evaluate_loss(
            loaded.model,
            dataset,
            batch_size=config.eval_batch_size,
            device=config.device,
            max_tokens=config.eval_max_tokens,
        )
        eval_result_dict = eval_result.to_dict()

    samples: list[GenerationSample] = []
    for i, prompt in enumerate(config.prompts):
        # Each prompt gets a distinct-but-deterministic seed (base seed +
        # index) so a multi-prompt report doesn't sample the same
        # continuation for every entry, while remaining fully reproducible
        # from the recorded config alone.
        settings = dataclasses.replace(config.generation, seed=config.generation.seed + i)
        prompt_ids = tokenizer.encode(prompt, allow_special=False)
        if not prompt_ids:
            bos = tokenizer.special_token_ids.get("<|bos|>")
            if bos is None:
                raise ValueError(
                    f"prompt {prompt!r} encoded to zero tokens and the tokenizer has no "
                    "'<|bos|>' special token to fall back to"
                )
            prompt_ids = [bos]
        generated_ids = generate(
            loaded.model,
            prompt_ids,
            settings,
            context_length=loaded.model_config.context_length,
            device=config.device,
        )
        samples.append(
            GenerationSample(
                prompt=prompt,
                generated_text=tokenizer.decode(generated_ids),
                settings=settings,
            )
        )

    report = SampleReport(
        checkpoint_path=str(config.checkpoint_path),
        step=loaded.step,
        model_config=dataclasses.asdict(loaded.model_config),
        eval_result=eval_result_dict,
        samples=samples,
    )
    json_path, md_path = save_report(report, config.output_path)

    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "step": loaded.step,
        "num_samples": len(samples),
        "eval_result": eval_result_dict,
    }
