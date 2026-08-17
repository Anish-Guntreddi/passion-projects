"""Throughput / peak-memory / val-loss-vs-tokens benchmark harness (FR10).

Component boundary: this module owns *measurement*, not model
correctness. It reuses ``forgelm.training.loop.Trainer`` to run real
optimizer steps (the thing being measured *is* the real training step --
reimplementing a second, parallel "benchmark step" would risk measuring
something subtly different from what ``forgelm train`` actually does) and
``forgelm.model``/``forgelm.dataset`` to build the model/data, but
contains no assertions about model correctness itself (those live in
``tests/unit/test_model_*.py`` and ``tests/integration/test_model_overfit.py``,
per spec §1.7 "benchmark code separated from model-correctness code").

Methodology (see ``docs/decisions/0010-phase5-compute-budget.md`` and
``benchmarks/methodology.md`` for the full write-up):

1. ``bench_warmup_steps`` untimed steps run first -- these absorb CUDA
   kernel selection, cuDNN autotune, and (when ``torch_compile=True``)
   ``torch.compile`` tracing/compilation cost, none of which should count
   toward steady-state throughput.
2. Peak-memory tracking resets right after warmup
   (``torch.cuda.reset_peak_memory_stats``), so the reported peak reflects
   only the measured window.
3. Exactly ``measured_steps`` further steps are timed individually
   (``torch.cuda.synchronize()`` before/after each step on CUDA, so GPU
   kernels are fully accounted for rather than only the CPU-side launch
   time).
4. Validation loss is sampled periodically during the measured window
   (every ``eval_interval`` steps, default: once at the end) via
   ``Trainer.evaluate()`` -- *not* timed as part of throughput, so
   occasional evaluation passes never skew the tokens/sec number.

CPU fallback (FR10 "CPU-fallback behavior"): every step above degrades
gracefully when ``device.type == "cpu"`` -- no CUDA calls are made, and
``peak_memory_bytes`` is reported as ``None`` rather than a fabricated or
crashing value.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forgelm.benchmarks.hardware import hardware_info, software_info
from forgelm.config import ModelConfig, TrainingConfig
from forgelm.dataset import NextTokenDataset
from forgelm.model import TransformerDecoder, count_parameters
from forgelm.seed import set_seed
from forgelm.training import Trainer


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Everything one :func:`run_benchmark` call measures, self-describing
    enough to be committed as a standalone JSON artifact.

    Per the spec's Part 3 benchmark plan ("every result records hardware,
    software versions, dtype, seed, model config, dataset, warmup policy,
    sample count"), ``train_dataset_path``/``val_dataset_path`` make the
    dataset identity recoverable from a single result file in isolation --
    not only from the surrounding methodology prose.
    """

    name: str
    param_count: int
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    train_dataset_path: str
    val_dataset_path: str
    device: str
    tokens_per_step: int
    bench_warmup_steps: int
    measured_steps: int
    total_step_time_s: float
    step_time_ms_mean: float
    step_time_ms_p50: float
    tokens_per_sec: float
    peak_memory_bytes: int | None
    final_train_loss: float
    val_loss: float
    val_loss_history: list[dict[str, float | int]]
    torch_compile: bool
    compile_error: str | None
    seed: int
    hardware: dict[str, Any]
    software: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _load_dataset(path: str | Path, context_length: int) -> NextTokenDataset:
    tokens = np.load(path)
    return NextTokenDataset(tokens.tolist(), context_length)


def run_benchmark(
    name: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_tokens_path: str | Path,
    val_tokens_path: str | Path,
    *,
    bench_warmup_steps: int = 10,
    measured_steps: int = 50,
    eval_interval: int | None = None,
    torch_compile: bool = False,
) -> BenchmarkResult:
    """Run one throughput/memory/val-loss measurement. See module docstring
    for the exact methodology.

    ``eval_interval=None`` (default) evaluates only once, after the final
    measured step; an explicit value builds a multi-point
    ``val_loss_history`` (tokens-trained -> val_loss curve).
    """
    if bench_warmup_steps < 0:
        raise ValueError(f"bench_warmup_steps must be non-negative, got {bench_warmup_steps}")
    if measured_steps <= 0:
        raise ValueError(f"measured_steps must be positive, got {measured_steps}")

    set_seed(training_config.seed)
    train_ds = _load_dataset(train_tokens_path, model_config.context_length)
    val_ds = _load_dataset(val_tokens_path, model_config.context_length)

    model = TransformerDecoder(model_config)
    param_count = count_parameters(model)

    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        model_config,
        training_config,
        dataset_paths=(str(train_tokens_path), str(val_tokens_path)),
    )

    compile_error: str | None = None
    compiled = False
    if torch_compile:
        try:
            trainer.model = torch.compile(trainer.model)  # type: ignore[assignment]
            compiled = True
        except Exception as exc:  # pragma: no cover - depends on local torch/triton build
            compile_error = f"{type(exc).__name__}: {exc}"

    device = trainer.device
    is_cuda = device.type == "cuda"

    # -- warmup (untimed) --------------------------------------------------
    for _ in range(bench_warmup_steps):
        trainer.train_step()

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    tokens_per_step = (
        training_config.micro_batch_size
        * training_config.grad_accum_steps
        * model_config.context_length
    )
    eval_every = eval_interval if eval_interval is not None else measured_steps

    step_times_s: list[float] = []
    val_loss_history: list[dict[str, float | int]] = []
    last_train_loss = float("nan")

    for i in range(1, measured_steps + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        metrics = trainer.train_step()
        if is_cuda:
            torch.cuda.synchronize(device)
        step_times_s.append(time.perf_counter() - start)
        last_train_loss = metrics.loss

        if i % eval_every == 0 or i == measured_steps:
            val_loss = trainer.evaluate()
            val_loss_history.append(
                {
                    "step": trainer.step,
                    "tokens": trainer.step * tokens_per_step,
                    "val_loss": val_loss,
                }
            )

    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if is_cuda else None

    total_time_s = sum(step_times_s)
    mean_step_s = total_time_s / len(step_times_s)
    sorted_times = sorted(step_times_s)
    p50_step_s = sorted_times[len(sorted_times) // 2]
    tokens_per_sec = tokens_per_step / mean_step_s

    return BenchmarkResult(
        name=name,
        param_count=param_count,
        model_config=dataclasses.asdict(model_config),
        training_config=dataclasses.asdict(training_config),
        train_dataset_path=str(train_tokens_path),
        val_dataset_path=str(val_tokens_path),
        device=str(device),
        tokens_per_step=tokens_per_step,
        bench_warmup_steps=bench_warmup_steps,
        measured_steps=measured_steps,
        total_step_time_s=total_time_s,
        step_time_ms_mean=mean_step_s * 1000.0,
        step_time_ms_p50=p50_step_s * 1000.0,
        tokens_per_sec=tokens_per_sec,
        peak_memory_bytes=peak_memory_bytes,
        final_train_loss=last_train_loss,
        val_loss=float(val_loss_history[-1]["val_loss"])
        if val_loss_history
        else trainer.evaluate(),
        val_loss_history=val_loss_history,
        torch_compile=compiled,
        compile_error=compile_error,
        seed=training_config.seed,
        hardware=hardware_info(device),
        software=software_info(),
        timestamp=datetime.now(UTC).isoformat(),
    )
