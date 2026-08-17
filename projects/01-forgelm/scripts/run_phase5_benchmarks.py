#!/usr/bin/env python3
"""Phase 5 scaling-experiment driver.

The single source of truth for the whole experiment matrix described in
``docs/decisions/0010-phase5-compute-budget.md``: three model-size
configs (small/medium/larger), a context-length sweep, a batch-size
sweep, and one optional torch.compile A/B, all run through
``forgelm.benchmarks.harness.run_benchmark`` against the already-built
``examples/alice_in_wonderland.txt`` dataset artifacts.

A driver *script* (not ten hand-written YAML configs, though
``forgelm benchmark run --config ...`` -- see ``configs/benchmark_example.yaml``
-- remains the config-driven single-run entry point FR1 asks for) because
a parameter sweep is naturally one small piece of code, not N nearly-
identical files to keep in sync by hand.

Usage (from an activated .venv, project root):

    python scripts/run_phase5_benchmarks.py --device cuda

Writes one JSON file per run to ``benchmarks/results/``, plus
``phase5_summary.json`` (every result, for the plotting script) and
``phase5_manifest.json`` (total wall-clock time + environment info, the
evidence that the D8 compute budget in ADR 0010 was respected).

Hard rule (repo-wide, see the orchestrator's GPU-benchmark-lock protocol):
this script's real (--device cuda) invocation must only run while holding
the repo's GPU benchmark lock, released immediately after it exits. This
script itself does not take the lock -- that is deliberately kept outside
project source (it is a cross-project, repo-root protocol, not something
``forgelm`` should embed) and is the caller's responsibility.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from forgelm.benchmarks import BenchmarkResult, run_benchmark
from forgelm.config import ModelConfig, TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "artifacts" / "dataset"
TRAIN_TOKENS_PATH = DATASET_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = DATASET_DIR / "val_tokens.npy"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

# vocab_size must match the tokenizer the dataset artifacts were built
# with -- see configs/tokenizer_train.yaml.
VOCAB_SIZE = 512
DATASET_CONTEXT_LENGTH = 128  # matches configs/dataset_build.yaml


def _training_config(
    *, micro_batch_size: int, total_steps: int, seed: int = 1337
) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=1,
        max_steps=total_steps,
        warmup_steps=min(20, total_steps),
        lr=3e-4,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip_norm=1.0,
        eval_interval=total_steps,  # unused by the harness's own eval loop
        eval_batches=5,
        device="cuda",
        precision="auto",
    )


def _model_config(
    *, d_model: int, n_layers: int, n_heads: int, d_ff: int, context_length: int
) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        tie_weights=True,
        dtype="float32",
    )


# -- the three scaling configs (ADR 0010) -----------------------------------------

SCALING_CONFIGS: dict[str, ModelConfig] = {
    "small": _model_config(
        d_model=64, n_layers=2, n_heads=4, d_ff=256, context_length=DATASET_CONTEXT_LENGTH
    ),
    "medium": _model_config(
        d_model=128, n_layers=4, n_heads=4, d_ff=512, context_length=DATASET_CONTEXT_LENGTH
    ),
    "larger": _model_config(
        d_model=256, n_layers=6, n_heads=8, d_ff=1024, context_length=DATASET_CONTEXT_LENGTH
    ),
}

SCALING_MICRO_BATCH_SIZE = 16
SCALING_BENCH_WARMUP_STEPS = 10
SCALING_MEASURED_STEPS = 300
SCALING_EVAL_INTERVAL = 30  # 10 points on the val-loss-vs-tokens curve

CONTEXT_LENGTH_SWEEP = [32, 64, 128, 256]
BATCH_SIZE_SWEEP = [4, 8, 16, 32, 64]
SWEEP_BENCH_WARMUP_STEPS = 5
SWEEP_MEASURED_STEPS = 20

COMPILE_AB_BENCH_WARMUP_STEPS = 20  # higher: amortizes torch.compile's own tracing cost
COMPILE_AB_MEASURED_STEPS = 50


def _save(result: BenchmarkResult, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_and_report(
    name: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    bench_warmup_steps: int,
    measured_steps: int,
    eval_interval: int | None = None,
    torch_compile: bool = False,
) -> BenchmarkResult:
    print(f"== {name} ==", flush=True)
    print(
        f"   d_model={model_config.d_model} n_layers={model_config.n_layers} "
        f"context_length={model_config.context_length} "
        f"batch={training_config.micro_batch_size} warmup={bench_warmup_steps} "
        f"measured={measured_steps} compile={torch_compile}",
        flush=True,
    )
    result = run_benchmark(
        name,
        model_config,
        training_config,
        TRAIN_TOKENS_PATH,
        VAL_TOKENS_PATH,
        bench_warmup_steps=bench_warmup_steps,
        measured_steps=measured_steps,
        eval_interval=eval_interval,
        torch_compile=torch_compile,
    )
    path = _save(result, name)
    print(
        f"   -> params={result.param_count:,} tokens/sec={result.tokens_per_sec:,.0f} "
        f"peak_mem={result.peak_memory_bytes} val_loss={result.val_loss:.4f} "
        f"-> {path.relative_to(PROJECT_ROOT)}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--skip-compile-ab",
        action="store_true",
        help="Skip the optional torch.compile A/B (spec Part 3 marks it optional).",
    )
    args = parser.parse_args()

    if not TRAIN_TOKENS_PATH.exists() or not VAL_TOKENS_PATH.exists():
        raise SystemExit(
            f"dataset artifacts not found at {DATASET_DIR} -- run:\n"
            "  forgelm tokenizer train --config configs/tokenizer_train.yaml\n"
            "  forgelm dataset build --config configs/dataset_build.yaml\n"
            "first."
        )

    wall_start = time.perf_counter()
    all_results: list[dict[str, object]] = []

    # -- 1. model-size scaling (tokens/sec vs. model size, val loss vs. tokens) --
    for name, model_config in SCALING_CONFIGS.items():
        training_config = _training_config(
            micro_batch_size=SCALING_MICRO_BATCH_SIZE,
            total_steps=SCALING_BENCH_WARMUP_STEPS + SCALING_MEASURED_STEPS,
        )
        training_config = dataclasses.replace(training_config, device=args.device)
        result = _run_and_report(
            f"scaling_{name}",
            model_config,
            training_config,
            bench_warmup_steps=SCALING_BENCH_WARMUP_STEPS,
            measured_steps=SCALING_MEASURED_STEPS,
            eval_interval=SCALING_EVAL_INTERVAL,
        )
        all_results.append(result.to_dict())

    # -- 2. peak memory vs. context length (medium config, batch size fixed) -----
    for context_length in CONTEXT_LENGTH_SWEEP:
        model_config = _model_config(
            d_model=128, n_layers=4, n_heads=4, d_ff=512, context_length=context_length
        )
        training_config = _training_config(
            micro_batch_size=SCALING_MICRO_BATCH_SIZE,
            total_steps=SWEEP_BENCH_WARMUP_STEPS + SWEEP_MEASURED_STEPS,
        )
        training_config = dataclasses.replace(training_config, device=args.device)
        result = _run_and_report(
            f"context_length_{context_length}",
            model_config,
            training_config,
            bench_warmup_steps=SWEEP_BENCH_WARMUP_STEPS,
            measured_steps=SWEEP_MEASURED_STEPS,
        )
        all_results.append(result.to_dict())

    # -- 3. peak memory vs. batch size (medium config, context length fixed) -----
    for batch_size in BATCH_SIZE_SWEEP:
        model_config = SCALING_CONFIGS["medium"]
        training_config = _training_config(
            micro_batch_size=batch_size,
            total_steps=SWEEP_BENCH_WARMUP_STEPS + SWEEP_MEASURED_STEPS,
        )
        training_config = dataclasses.replace(training_config, device=args.device)
        result = _run_and_report(
            f"batch_size_{batch_size}",
            model_config,
            training_config,
            bench_warmup_steps=SWEEP_BENCH_WARMUP_STEPS,
            measured_steps=SWEEP_MEASURED_STEPS,
        )
        all_results.append(result.to_dict())

    # -- 4. optional torch.compile A/B (small config) -----------------------------
    if not args.skip_compile_ab:
        for compile_flag in (False, True):
            model_config = SCALING_CONFIGS["small"]
            training_config = _training_config(
                micro_batch_size=SCALING_MICRO_BATCH_SIZE,
                total_steps=COMPILE_AB_BENCH_WARMUP_STEPS + COMPILE_AB_MEASURED_STEPS,
            )
            training_config = dataclasses.replace(training_config, device=args.device)
            suffix = "compiled" if compile_flag else "eager"
            result = _run_and_report(
                f"compile_ab_{suffix}",
                model_config,
                training_config,
                bench_warmup_steps=COMPILE_AB_BENCH_WARMUP_STEPS,
                measured_steps=COMPILE_AB_MEASURED_STEPS,
                torch_compile=compile_flag,
            )
            all_results.append(result.to_dict())

    wall_elapsed_s = time.perf_counter() - wall_start

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase5_summary.json").write_text(
        json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "num_runs": len(all_results),
        "total_wall_clock_s": wall_elapsed_s,
        "total_wall_clock_human": f"{wall_elapsed_s / 60:.2f} min",
        "device": args.device,
        "skip_compile_ab": args.skip_compile_ab,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_names": [r["name"] for r in all_results],
    }
    (RESULTS_DIR / "phase5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n{len(all_results)} runs complete in {wall_elapsed_s / 60:.2f} min total.", flush=True)
    print(f"Summary: {(RESULTS_DIR / 'phase5_summary.json').relative_to(PROJECT_ROOT)}", flush=True)
    print(
        f"Manifest: {(RESULTS_DIR / 'phase5_manifest.json').relative_to(PROJECT_ROOT)}", flush=True
    )


if __name__ == "__main__":
    main()
