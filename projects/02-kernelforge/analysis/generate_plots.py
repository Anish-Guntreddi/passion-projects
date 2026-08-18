#!/usr/bin/env python3
"""Phase 7 analysis pipeline (FR7): reads committed benchmarks/raw/*.jsonl
and profiling/occupancy/gemm.jsonl records and writes PNG plots to
benchmarks/plots/. Never invents a number -- every plotted value is read
directly from a committed, schema-validated record (benchmarks/schema/
bench_result.schema.json / profiling/schema/occupancy_report.schema.json).
CUDA implementation stays C++ throughout (analysis/ never contains kernel
code, per the spec's FR7 wording) -- this file only aggregates/plots.

Usage (from repo root, after `python3 -m venv analysis/.venv && \
    analysis/.venv/bin/pip install -r analysis/requirements.txt`):
    analysis/.venv/bin/python analysis/generate_plots.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless -- this script never opens a display window
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "benchmarks" / "raw"
OCC = ROOT / "profiling" / "occupancy"
OUT = ROOT / "benchmarks" / "plots"

# Validated categorical palette (dataviz skill, references/palette.md) --
# fixed hue order, used identically as "series N" across every chart below
# so the same rung/variant always gets the same color everywhere.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK_PRIMARY,
        "grid.color": GRIDLINE,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_jsonl(path: pathlib.Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def style_axes(ax) -> None:
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_transpose(out_dir: pathlib.Path) -> None:
    df = load_jsonl(RAW / "transpose.jsonl")
    df["side"] = df["rows"]  # square matrices only in this sweep
    piv = df.pivot_table(index="side", columns="variant", values="effective_bandwidth_gb_s")
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(piv.index, piv["v1_naive"], marker="o", color=PALETTE[0], linewidth=2, label="v1_naive (pathological write)")
    ax.plot(piv.index, piv["v2_tiled"], marker="o", color=PALETTE[1], linewidth=2, label="v2_tiled (coalesced both ways)")
    ax.axhline(1008.1, color=BASELINE, linestyle="--", linewidth=1, label="1008.1 GB/s theoretical peak")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("matrix side (n x n)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Phase 1: Transpose -- naive vs. shared-memory-tiled")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "transpose_bandwidth.png", dpi=150)
    plt.close(fig)


def plot_reduction(out_dir: pathlib.Path) -> None:
    df = load_jsonl(RAW / "reduction.jsonl")
    variants = [
        ("v0_naive_global_atomic", "V0 naive_global_atomic", PALETTE[7]),
        ("v1_naive_interleaved", "V1 naive_interleaved", PALETTE[0]),
        ("v2_sequential_addressing", "V2 sequential_addressing", PALETTE[1]),
        ("v3_warp_shuffle", "V3 warp_shuffle", PALETTE[2]),
        ("v4_vectorized_coarsened", "V4 vectorized_coarsened", PALETTE[3]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant, label, color in variants:
        sub = df[df["variant"] == variant].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"], sub["effective_bandwidth_gb_s"], marker="o", color=color, linewidth=2, label=label)
    ax.axhline(1008.1, color=BASELINE, linestyle="--", linewidth=1, label="1008.1 GB/s theoretical peak")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (elements)")
    ax.set_ylabel("effective bandwidth (GB/s, log scale)")
    ax.set_title("Phase 2: Reduction ladder (V0-V4)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reduction_ladder.png", dpi=150)
    plt.close(fig)


def plot_histogram(out_dir: pathlib.Path) -> None:
    uni = load_jsonl(RAW / "histogram_uniform.jsonl")
    skew = load_jsonl(RAW / "histogram_skewed.jsonl")
    n_max = uni["n"].max()
    uni = uni[uni["n"] == n_max]
    skew = skew[skew["n"] == n_max]

    variants = ["v1_global_atomic", "v2_privatized", "v3_privatized_coarsened"]
    labels = ["V1 global_atomic", "V2 privatized", "V3 privatized_coarsened"]
    uni_vals = [uni[uni["variant"] == v]["effective_bandwidth_gb_s"].iloc[0] for v in variants]
    skew_vals = [skew[skew["variant"] == v]["effective_bandwidth_gb_s"].iloc[0] for v in variants]

    x = range(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar([i - width / 2 for i in x], uni_vals, width, color=PALETTE[0], label="uniform (low contention)")
    ax.bar([i + width / 2 for i in x], skew_vals, width, color=PALETTE[1], label="skewed (high contention)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title(f"Phase 3: Histogram -- contention sensitivity (n={n_max:,})")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "histogram_contention.png", dpi=150)
    plt.close(fig)


def plot_gemm(out_dir: pathlib.Path) -> None:
    df = load_jsonl(RAW / "gemm.jsonl")
    variants = [
        ("v1_naive", "V1 naive", PALETTE[0]),
        ("v2_coalesced", "V2 coalesced", PALETTE[1]),
        ("v3_tiled", "V3 tiled", PALETTE[2]),
        ("v4_register_tiled", "V4 register_tiled", PALETTE[3]),
        ("cublas_sgemm_ceiling", "cuBLAS (ceiling, not a rung)", PALETTE[7]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant, label, color in variants:
        sub = df[df["variant"] == variant].sort_values("rows")
        if sub.empty:
            continue
        style = "--" if variant == "cublas_sgemm_ceiling" else "-"
        ax.plot(sub["rows"], sub["gflops"], marker="o", color=color, linewidth=2, linestyle=style, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_title("Phase 4: GEMM ladder (V1-V4) vs. cuBLAS ceiling")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "gemm_ladder.png", dpi=150)
    plt.close(fig)


def plot_softmax_rmsnorm(out_dir: pathlib.Path) -> None:
    softmax = load_jsonl(RAW / "softmax.jsonl")
    rmsnorm = load_jsonl(RAW / "rmsnorm.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, df, title, variants in (
        (axes[0], softmax, "Softmax ladder", [("v1_naive", "V1 naive"), ("v2_warp_shuffle", "V2 warp_shuffle"), ("v3_fused_online", "V3 fused_online")]),
        (axes[1], rmsnorm, "RMSNorm ladder", [("v1_naive", "V1 naive"), ("v2_warp_shuffle", "V2 warp_shuffle"), ("v3_vectorized", "V3 vectorized")]),
    ):
        for (variant, label), color in zip(variants, PALETTE):
            sub = df[df["variant"] == variant].sort_values("cols")
            if sub.empty:
                continue
            ax.plot(sub["cols"], sub["stats_ms"].apply(lambda s: s["median"]), marker="o", color=color, linewidth=2, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("cols (row width)")
        ax.set_ylabel("median kernel time (ms, log scale)")
        ax.set_title(title)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 5: Softmax and RMSNorm ladders", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "softmax_rmsnorm_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gemm_occupancy_crossover(out_dir: pathlib.Path) -> None:
    """Phase 6: grid_utilization_fraction (occupancy evidence) side-by-side
    with the wall-clock V4/V3 GFLOP/s ratio (benchmark evidence) -- two
    panels sharing an x-axis, never a dual-y-axis single panel (dataviz
    skill: "never a dual-axis chart").

    profiling/occupancy/gemm.jsonl and benchmarks/raw/gemm.jsonl are two
    independently-generated files -- this never assumes their record order
    matches. Every x-axis size plotted here is read from that record's own
    problem_m/problem_n (occupancy) or rows/cols (bench) fields and joined
    by VALUE (pd.merge), so a re-run that drops, reorders, or adds a record
    fails loudly (via the asserts below) instead of silently mislabeling
    the x-axis.
    """
    occ = load_jsonl(OCC / "gemm.jsonl")
    bench = load_jsonl(RAW / "gemm.jsonl")

    def occ_side(variant: str, value_col: str) -> pd.DataFrame:
        sub = occ[occ["variant"] == variant]
        assert (sub["problem_m"] > 0).all() and (sub["problem_n"] > 0).all(), (
            f"gemm.jsonl variant={variant!r} has a record with problem_m/problem_n <= 0 "
            "(not applicable/not requested) -- this plot requires every gemm occupancy "
            "record to carry the problem shape it was queried against."
        )
        assert (sub["problem_m"] == sub["problem_n"]).all(), (
            f"gemm.jsonl variant={variant!r} has a non-square (problem_m != problem_n) "
            "record -- this plot assumes the M=N=K sweep documented in "
            "scripts/run_occupancy_report.sh."
        )
        return sub[["problem_m", "grid_utilization_fraction"]].rename(
            columns={"grid_utilization_fraction": value_col}
        )

    occ_merged = occ_side("v3_tiled", "v3_util").merge(
        occ_side("v4_register_tiled", "v4_util"), on="problem_m", how="inner", validate="one_to_one"
    ).sort_values("problem_m").reset_index(drop=True)
    assert not occ_merged.empty, "gemm.jsonl: no matching problem_m between v3_tiled and v4_register_tiled"
    sizes = occ_merged["problem_m"].tolist()

    def bench_side(variant: str, value_col: str) -> pd.DataFrame:
        sub = bench[bench["variant"] == variant]
        assert (sub["rows"] == sub["cols"]).all(), (
            f"gemm.jsonl variant={variant!r} has a non-square (rows != cols) benchmark "
            "record -- expected the M=N=K sweep this plot documents."
        )
        return sub[["rows", "gflops"]].rename(columns={"rows": "problem_m", "gflops": value_col})

    bench_merged = bench_side("v3_tiled", "v3_gflops").merge(
        bench_side("v4_register_tiled", "v4_gflops"), on="problem_m", how="inner", validate="one_to_one"
    )
    bench_merged = bench_merged[bench_merged["problem_m"].isin(sizes)].sort_values("problem_m").reset_index(drop=True)
    assert bench_merged["problem_m"].tolist() == sizes, (
        "profiling/occupancy/gemm.jsonl and benchmarks/raw/gemm.jsonl do not cover the "
        f"same problem-size sweep: occupancy sizes={sizes} bench sizes={bench_merged['problem_m'].tolist()} "
        "-- refusing to plot a mislabeled x-axis."
    )
    ratio = (bench_merged["v4_gflops"] / bench_merged["v3_gflops"]).to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(sizes, occ_merged["v3_util"] * 100, marker="o", color=PALETTE[2], linewidth=2, label="V3 tiled (32x32 tile)")
    ax1.plot(sizes, occ_merged["v4_util"] * 100, marker="o", color=PALETTE[3], linewidth=2, label="V4 register_tiled (64x64 tile)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("M = N = K")
    ax1.set_ylabel("grid utilization (%)")
    ax1.set_title("Grid-level utilization (occupancy_report)")
    style_axes(ax1)
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(sizes, ratio, marker="o", color=PALETTE[0], linewidth=2)
    ax2.axhline(1.0, color=BASELINE, linestyle="--", linewidth=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("M = N = K")
    ax2.set_ylabel("V4/V3 GFLOP/s ratio")
    ax2.set_title("Wall-clock speedup (benchmarks/raw/gemm.jsonl)")
    style_axes(ax2)

    fig.suptitle(
        "Phase 6 case study 3: grid utilization crosses 100% at the same size (1024)\nwall-clock crosses from regression to decisive win",
        y=1.06,
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gemm_occupancy_crossover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_transpose(OUT)
    plot_reduction(OUT)
    plot_histogram(OUT)
    plot_gemm(OUT)
    plot_softmax_rmsnorm(OUT)
    plot_gemm_occupancy_crossover(OUT)
    written = sorted(p.name for p in OUT.glob("*.png"))
    print(f"generate_plots: wrote {len(written)} PNG(s) to {OUT}:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
