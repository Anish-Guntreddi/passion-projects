"""Sample-report format (FR9): one checkpoint's loss/perplexity plus a set
of documented-decoding-settings generations, as a single artifact.

Written as JSON (machine-readable, the source of truth) and rendered to
Markdown (human-readable, what a reviewer actually reads -- spec §1.6
"sample generations with documented decoding settings"). Both are derived
from the same :class:`SampleReport` so they can never drift apart.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from forgelm.config import GenerationSettings


@dataclasses.dataclass(frozen=True)
class GenerationSample:
    """One prompt's generation, with the exact settings that produced it."""

    prompt: str
    generated_text: str
    settings: GenerationSettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "settings": dataclasses.asdict(self.settings),
        }


@dataclasses.dataclass(frozen=True)
class SampleReport:
    """A checkpoint's evaluation result plus its documented generations."""

    checkpoint_path: str
    step: int
    model_config: dict[str, Any]
    eval_result: dict[str, Any] | None
    samples: list[GenerationSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "step": self.step,
            "model_config": self.model_config,
            "eval_result": self.eval_result,
            "samples": [s.to_dict() for s in self.samples],
        }


def render_markdown(report: SampleReport) -> str:
    """Human-readable rendering of ``report`` (spec §1.6 deliverable)."""
    lines: list[str] = []
    lines.append("# Sample report")
    lines.append("")
    lines.append(f"- Checkpoint: `{report.checkpoint_path}`")
    lines.append(f"- Training step: {report.step}")
    mc = report.model_config
    lines.append(
        f"- Model: d_model={mc.get('d_model')}, n_layers={mc.get('n_layers')}, "
        f"n_heads={mc.get('n_heads')}, d_ff={mc.get('d_ff')}, "
        f"context_length={mc.get('context_length')}, vocab_size={mc.get('vocab_size')}"
    )
    lines.append("")

    if report.eval_result is not None:
        er = report.eval_result
        lines.append("## Evaluation")
        lines.append("")
        lines.append(f"- Loss: {er['loss']:.4f}")
        lines.append(f"- Perplexity: {er['perplexity']:.4f}")
        lines.append(f"- Tokens evaluated: {er['tokens_evaluated']}")
        lines.append(f"- Batches: {er['num_batches']}")
        lines.append("")

    lines.append("## Generations")
    lines.append("")
    for i, sample in enumerate(report.samples, start=1):
        s = sample.settings
        mode = "greedy" if not s.do_sample else "sampling"
        decoding_bits = [f"mode={mode}"]
        if s.do_sample:
            decoding_bits.append(f"temperature={s.temperature}")
            if s.top_k is not None:
                decoding_bits.append(f"top_k={s.top_k}")
            if s.top_p is not None:
                decoding_bits.append(f"top_p={s.top_p}")
            decoding_bits.append(f"seed={s.seed}")
        decoding_bits.append(f"max_new_tokens={s.max_new_tokens}")

        lines.append(f"### Sample {i}")
        lines.append("")
        lines.append(f"Decoding settings: `{', '.join(decoding_bits)}`")
        lines.append("")
        lines.append(f"**Prompt:** {sample.prompt!r}")
        lines.append("")
        lines.append("**Generated:**")
        lines.append("")
        lines.append("```")
        lines.append(sample.generated_text)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def save_report(report: SampleReport, path: str | Path) -> tuple[Path, Path]:
    """Write ``report`` as ``<path>.json`` and ``<path>.md`` (``path``'s
    suffix, if any, is replaced). Returns ``(json_path, md_path)``."""
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
