"""Unit tests for forgelm.benchmarks.harness.run_benchmark, run on CPU with
tiny models/step counts -- fast correctness tests, not the real measured
Phase 5 numbers (those come from real GPU runs under the repo's
GPU-benchmark lock protocol; see scripts/run_phase5_benchmarks.py and
benchmarks/methodology.md). Exercises FR10's "CPU-fallback behavior"
directly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgelm.benchmarks import run_benchmark
from forgelm.config import ModelConfig, TrainingConfig
from forgelm.model import expected_parameter_count


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=32, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)


def _tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        micro_batch_size=4,
        max_steps=1000,
        warmup_steps=2,
        eval_interval=1000,
        eval_batches=1,
        device="cpu",
        precision="fp32",
    )


def _write_token_arrays(tmp_path: Path) -> tuple[Path, Path]:
    train = np.array([i % 32 for i in range(500)], dtype=np.int64)
    val = np.array([i % 32 for i in range(200)], dtype=np.int64)
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    np.save(train_path, train)
    np.save(val_path, val)
    return train_path, val_path


def test_run_benchmark_on_cpu_reports_no_peak_memory(tmp_path: Path) -> None:
    """FR10 'CPU-fallback behavior': peak_memory_bytes must be None, not a
    crash or a fabricated value, when no CUDA device is used."""
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "cpu-smoke",
        _tiny_model_config(),
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=2,
        measured_steps=3,
    )
    assert result.peak_memory_bytes is None
    assert result.device == "cpu"


def test_run_benchmark_reports_sane_throughput_fields(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    train_path, val_path = _write_token_arrays(tmp_path)

    result = run_benchmark(
        "cpu-throughput",
        model_config,
        training_config,
        train_path,
        val_path,
        bench_warmup_steps=1,
        measured_steps=4,
    )

    assert result.name == "cpu-throughput"
    assert result.measured_steps == 4
    assert result.tokens_per_step == training_config.micro_batch_size * model_config.context_length
    assert result.tokens_per_sec > 0
    assert result.step_time_ms_mean > 0
    assert result.total_step_time_s > 0
    assert result.final_train_loss == result.final_train_loss  # not NaN


def test_run_benchmark_param_count_matches_the_analytic_formula(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "param-count",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=0,
        measured_steps=1,
    )
    assert result.param_count == expected_parameter_count(model_config)


def test_run_benchmark_val_loss_history_length_matches_eval_interval(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "eval-curve",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=0,
        measured_steps=6,
        eval_interval=2,
    )
    # Evaluated at steps 2, 4, 6 -> 3 points.
    assert [p["step"] for p in result.val_loss_history] == [2, 4, 6]
    assert result.val_loss == result.val_loss_history[-1]["val_loss"]


def test_run_benchmark_default_eval_interval_evaluates_once_at_the_end(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "single-eval",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=0,
        measured_steps=5,
    )
    assert len(result.val_loss_history) == 1
    assert result.val_loss_history[0]["step"] == 5


def test_run_benchmark_records_dataset_identity(tmp_path: Path) -> None:
    """Spec Part 3: 'every result records ... dataset' -- a single result
    JSON must be able to name its dataset without relying on surrounding
    methodology prose."""
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "dataset-identity",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=0,
        measured_steps=1,
    )
    assert result.train_dataset_path == str(train_path)
    assert result.val_dataset_path == str(val_path)


def test_run_benchmark_records_hardware_and_software_info(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "env-record",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=0,
        measured_steps=1,
    )
    assert "torch_version" in result.software
    assert "python_version" in result.software
    assert "cpu" in result.hardware


def test_run_benchmark_rejects_non_positive_measured_steps(tmp_path: Path) -> None:
    train_path, val_path = _write_token_arrays(tmp_path)
    with pytest.raises(ValueError, match="measured_steps"):
        run_benchmark(
            "bad",
            _tiny_model_config(),
            _tiny_training_config(),
            train_path,
            val_path,
            measured_steps=0,
        )


def test_run_benchmark_torch_compile_either_succeeds_or_fails_gracefully(tmp_path: Path) -> None:
    """The optional torch.compile A/B path (spec Part 3 benchmark plan)
    must never crash the harness even if compilation itself fails on this
    machine's torch/triton build -- it should fall back to an uncompiled
    run and record why via ``compile_error``."""
    model_config = _tiny_model_config()
    train_path, val_path = _write_token_arrays(tmp_path)
    result = run_benchmark(
        "compile-ab",
        model_config,
        _tiny_training_config(),
        train_path,
        val_path,
        bench_warmup_steps=1,
        measured_steps=2,
        torch_compile=True,
    )
    if result.torch_compile:
        assert result.compile_error is None
    else:
        assert result.compile_error is not None
    assert result.tokens_per_sec > 0  # the run itself still completed either way


def test_run_benchmark_is_reproducible_given_the_same_seed(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    training_config = _tiny_training_config()
    train_path, val_path = _write_token_arrays(tmp_path)

    result_a = run_benchmark(
        "repro",
        model_config,
        training_config,
        train_path,
        val_path,
        bench_warmup_steps=1,
        measured_steps=3,
    )
    result_b = run_benchmark(
        "repro",
        model_config,
        training_config,
        train_path,
        val_path,
        bench_warmup_steps=1,
        measured_steps=3,
    )
    assert result_a.final_train_loss == pytest.approx(result_b.final_train_loss)
    assert result_a.val_loss == pytest.approx(result_b.val_loss)
