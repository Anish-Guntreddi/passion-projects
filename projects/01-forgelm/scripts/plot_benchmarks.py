#!/usr/bin/env python3
"""Render the Phase 5 benchmark plots (spec Part 3 benchmark plan) from
``benchmarks/results/phase5_summary.json`` -- produced by
``scripts/run_phase5_benchmarks.py``.

Every number plotted here is read directly from that committed JSON
artifact; this script computes no numbers of its own beyond simple
arithmetic (MiB conversion, sorting) on already-measured fields, per the
portfolio-wide "no benchmark number is ever fabricated" rule.

Usage (from an activated .venv with the `plots` extra installed --
`pip install -e ".[plots]"`):

    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display server in WSL/CI
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "phase5_summary.json"
PLOTS_DIR = PROJECT_ROOT / "benchmarks" / "plots"

_BYTES_PER_MIB = 1024 * 1024


def _load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found -- run scripts/run_phase5_benchmarks.py first.")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _by_prefix(results: list[dict], prefix: str) -> list[dict]:
    return [r for r in results if r["name"].startswith(prefix)]


def plot_tokens_per_sec_vs_model_size(results: list[dict]) -> Path:
    runs = sorted(_by_prefix(results, "scaling_"), key=lambda r: r["param_count"])
    params = [r["param_count"] for r in runs]
    tokens_per_sec = [r["tokens_per_sec"] for r in runs]
    labels = [r["name"].removeprefix("scaling_") for r in runs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(params, tokens_per_sec, marker="o")
    for p, t, label in zip(params, tokens_per_sec, labels, strict=True):
        ax.annotate(label, (p, t), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Tokens / sec")
    ax.set_xscale("log")
    ax.set_title("Throughput vs. model size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "tokens_per_sec_vs_model_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_peak_memory_vs_context_length(results: list[dict]) -> Path:
    runs = sorted(
        _by_prefix(results, "context_length_"),
        key=lambda r: r["model_config"]["context_length"],
    )
    context_lengths = [r["model_config"]["context_length"] for r in runs]
    peak_mem_mib = [(r["peak_memory_bytes"] or 0) / _BYTES_PER_MIB for r in runs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(context_lengths, peak_mem_mib, marker="o")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Peak GPU memory (MiB)")
    ax.set_title("Peak memory vs. context length (medium config)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "peak_memory_vs_context_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_peak_memory_vs_batch_size(results: list[dict]) -> Path:
    runs = sorted(
        _by_prefix(results, "batch_size_"),
        key=lambda r: r["training_config"]["micro_batch_size"],
    )
    batch_sizes = [r["training_config"]["micro_batch_size"] for r in runs]
    peak_mem_mib = [(r["peak_memory_bytes"] or 0) / _BYTES_PER_MIB for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(batch_sizes, peak_mem_mib, marker="o")
    ax.set_xlabel("Micro batch size")
    ax.set_ylabel("Peak GPU memory (MiB)")
    ax.set_title("Peak memory vs. batch size\n(medium config, context_length=128)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "peak_memory_vs_batch_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_val_loss_vs_tokens(results: list[dict]) -> Path:
    runs = _by_prefix(results, "scaling_")
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in sorted(runs, key=lambda r: r["param_count"]):
        history = r["val_loss_history"]
        tokens = [p["tokens"] for p in history]
        val_loss = [p["val_loss"] for p in history]
        label = f"{r['name'].removeprefix('scaling_')} ({r['param_count']:,} params)"
        ax.plot(tokens, val_loss, marker="o", label=label)
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss vs. training tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "val_loss_vs_tokens.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_compile_ab(results: list[dict]) -> Path | None:
    eager = next((r for r in results if r["name"] == "compile_ab_eager"), None)
    compiled = next((r for r in results if r["name"] == "compile_ab_compiled"), None)
    if eager is None or compiled is None:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    compiled_label = (
        "compiled\n(succeeded)"
        if compiled["compile_error"] is None
        else "compiled\n(fell back: error)"
    )
    names = ["eager", compiled_label]
    values = [eager["tokens_per_sec"], compiled["tokens_per_sec"]]
    ax.bar(names, values)
    ax.set_ylabel("Tokens / sec")
    ax.set_title("torch.compile A/B (small config)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "torch_compile_ab.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    results = _load_results()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    written = [
        plot_tokens_per_sec_vs_model_size(results),
        plot_peak_memory_vs_context_length(results),
        plot_peak_memory_vs_batch_size(results),
        plot_val_loss_vs_tokens(results),
    ]
    compile_plot = plot_compile_ab(results)
    if compile_plot is not None:
        written.append(compile_plot)

    for path in written:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
