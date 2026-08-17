from __future__ import annotations

import json
from pathlib import Path

from forgelm.training import MetricsLogger, StepMetrics


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_log_step_appends_one_json_line_with_step_fields(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log_step(StepMetrics(step=1, loss=2.5, lr=1e-3, grad_norm=0.7))

    records = _read_jsonl(path)
    assert records == [
        {"kind": "step", "step": 1, "loss": 2.5, "lr": 1e-3, "grad_norm": 0.7},
    ]


def test_log_eval_appends_one_json_line_with_val_loss(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log_eval(step=3, val_loss=4.2)

    records = _read_jsonl(path)
    assert records == [{"kind": "eval", "step": 3, "val_loss": 4.2}]


def test_multiple_calls_append_in_order(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log_step(StepMetrics(step=1, loss=3.0, lr=1e-3, grad_norm=1.0))
    logger.log_eval(step=1, val_loss=2.9)
    logger.log_step(StepMetrics(step=2, loss=2.8, lr=1e-3, grad_norm=0.9))

    records = _read_jsonl(path)
    assert [r["kind"] for r in records] == ["step", "eval", "step"]
    assert [r["step"] for r in records] == [1, 1, 2]


def test_metrics_logger_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log_eval(step=0, val_loss=1.0)
    assert path.exists()
