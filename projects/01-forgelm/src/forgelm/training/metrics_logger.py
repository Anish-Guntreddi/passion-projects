"""Per-step training metrics logging (FR6's "logging" requirement).

``Trainer.train_step()``/``Trainer.evaluate()`` compute a
:class:`~forgelm.training.loop.StepMetrics`/validation loss every call
but have no opinion on where those numbers go -- that I/O policy belongs
to whatever orchestrates the loop (``forgelm.cli.train_cli.run_training``),
matching this project's module boundary (``forgelm.training`` owns
optimization mechanics; *when* and *where* to persist a record of a run
is an orchestration concern, the same way ``run_training`` -- not
``Trainer`` -- decides the checkpoint filenames/interval).

:class:`MetricsLogger` is the minimal, dependency-free sink wired up
there: one JSON object per line appended to ``<output_dir>/metrics.jsonl``
(so the full loss/LR/grad-norm trajectory of a run can be reloaded and
plotted after the fact -- the data the Phase 5/6 "loss curves across
configurations" deliverable and the benchmark methodology need) plus a
human-readable line emitted via the stdlib ``logging`` module (to
stderr, not stdout, so it never corrupts a CLI command's JSON stdout
contract -- see ``tests/integration/test_cli_subprocess.py``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from forgelm.training.loop import StepMetrics

logger = logging.getLogger("forgelm.training")
if not logger.handlers:
    # Configured once, here, rather than left to the application entry
    # point: this project has no other logging call site, and scoping the
    # handler to this one named logger (never touching the root logger
    # via logging.basicConfig) keeps the change self-contained and inert
    # for every caller that never invokes MetricsLogger.log_step/log_eval
    # -- attaching a stderr handler produces no output on its own.
    _handler = logging.StreamHandler()  # defaults to sys.stderr
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class MetricsLogger:
    """Appends one JSON line per call to ``path`` and emits a log record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def log_step(self, metrics: StepMetrics) -> None:
        """Record one optimizer step's loss/lr/grad_norm."""
        self._append({"kind": "step", **dataclasses.asdict(metrics)})
        logger.info(
            "step=%d loss=%.4f lr=%.6g grad_norm=%.4f",
            metrics.step,
            metrics.loss,
            metrics.lr,
            metrics.grad_norm,
        )

    def log_eval(self, step: int, val_loss: float) -> None:
        """Record a validation pass's loss at a given training step."""
        self._append({"kind": "eval", "step": step, "val_loss": val_loss})
        logger.info("step=%d val_loss=%.4f", step, val_loss)
