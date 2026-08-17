"""Library-level benchmark operations the CLI wraps (also used by tests
and by scripts/run_phase5_benchmarks.py)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from forgelm.benchmarks import run_benchmark
from forgelm.config import BenchmarkRunConfig


def run_benchmark_cli(
    config: BenchmarkRunConfig, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Run one benchmark measurement and optionally write the result JSON.

    ``output_path`` overrides ``config.output_path`` when given (the CLI's
    ``--output`` flag), matching every other command's convention.
    """
    result = run_benchmark(
        config.name,
        config.model,
        config.training,
        config.train_tokens_path,
        config.val_tokens_path,
        bench_warmup_steps=config.bench_warmup_steps,
        measured_steps=config.measured_steps,
        eval_interval=config.eval_interval,
        torch_compile=config.torch_compile,
    )
    payload = dataclasses.asdict(result)

    resolved_output = output_path if output_path is not None else config.output_path
    if resolved_output is not None:
        resolved_output = Path(resolved_output)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return payload
