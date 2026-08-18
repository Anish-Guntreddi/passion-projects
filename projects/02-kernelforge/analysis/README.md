# Analysis Pipeline

Phase 7 ("Portfolio release") deliverable per spec FR7: a Python
(pandas/matplotlib) pipeline that turns committed `benchmarks/raw/*.jsonl`
(and, for Phase 6, `profiling/occupancy/gemm.jsonl`) records into plots
under `benchmarks/plots/`. CUDA implementation stays C++ throughout — this
directory never contains kernel code, only aggregation/plotting of
already-committed, schema-validated data.

## Setup

`pandas`/`matplotlib` are not available via `apt` without root on this
machine (no passwordless `sudo`), and installing into the system Python
directly is blocked (`error: externally-managed-environment`, PEP 668).
A local virtualenv sidesteps both:

```bash
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r analysis/requirements.txt
```

(`analysis/.venv/` is gitignored — regenerate it with the two commands
above rather than committing it.)

## Generating the plots

```bash
analysis/.venv/bin/python analysis/generate_plots.py
```

Writes 6 PNGs to `benchmarks/plots/`:

| File | Source data | Covers |
|---|---|---|
| `transpose_bandwidth.png` | `benchmarks/raw/transpose.jsonl` | Phase 1 §8 — naive vs. tiled coalescing |
| `reduction_ladder.png` | `benchmarks/raw/reduction.jsonl` | Phase 2 §9.1 — 5-rung reduction ladder |
| `histogram_contention.png` | `benchmarks/raw/histogram_{uniform,skewed}.jsonl` | Phase 3 §10 — contention sensitivity |
| `gemm_ladder.png` | `benchmarks/raw/gemm.jsonl` | Phase 4 §11 — 4-rung GEMM ladder vs. cuBLAS ceiling |
| `softmax_rmsnorm_ladder.png` | `benchmarks/raw/{softmax,rmsnorm}.jsonl` | Phase 5 §12-13 — both AI-primitive ladders |
| `gemm_occupancy_crossover.png` | `profiling/occupancy/gemm.jsonl` + `benchmarks/raw/gemm.jsonl` | Phase 6 case study 3 — grid-utilization crossover alongside the wall-clock speedup it explains |

Every plotted value is read directly from a committed, schema-validated
`.jsonl` record — nothing here is hand-entered or transcribed. Colors
follow a validated, colorblind-checked categorical palette (fixed hue
order per variant/rung, consistent across every chart) rather than
matplotlib's default cycle; `gemm_occupancy_crossover.png` uses two
side-by-side panels rather than a dual-y-axis single panel (never mix two
different-unit scales on one axis).

## Reading the data without this pipeline

Every `benchmarks/raw/*.jsonl` / `profiling/occupancy/*.jsonl` file is
directly loadable with the Python standard library alone, no dependency
needed:

```python
import json
records = [json.loads(line) for line in open("benchmarks/raw/transpose.jsonl")]
```

or with the pipeline's dependencies installed:

```python
import pandas as pd
df = pd.read_json("benchmarks/raw/transpose.jsonl", lines=True)
```
