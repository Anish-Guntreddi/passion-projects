# FlashLite

From naive attention to IO-aware fused attention. Full product spec:
`04-flashlite-spec.md` (repo root of `passion-projects`, one level above
this project). This README covers **Phase 0 (Math/reference harness)** and
**Phase 1 (Naive custom attention)** -- the phases implemented so far.

Target hardware: **NVIDIA GeForce RTX 4090, sm_89** (ADR 0001, matching
KernelForge's own target-architecture ADR) -- the only GPU this repo is
built and tested against. Build environment: **WSL2 Ubuntu on Windows 11**,
CUDA toolkit 12.6, Python 3.12, PyTorch 2.6.0+cu124.

## Status

| Phase | Deliverables | Exit criterion | Status |
|---|---|---|---|
| 0 | Reference attention, configurable causal masking, deterministic tensors, correctness comparator, benchmark result schema | Trusted PyTorch outputs exist across representative shapes | **Met** |
| 1 | Score computation, softmax, value aggregation as straightforward CUDA kernels | Custom path matches reference within documented tolerance | **Met** |

See `docs/decisions/` for the spec's open decisions D1-D4, D6, D7 (D5 is
explicitly deferred to Phase 6, resolved empirically). See
`docs/attention-math.md` for the V0/V1 math derivation and materialized-form
cost analysis, and `docs/architecture.md` for the data-flow/package-layout
overview. See `benchmarks/methodology.md` SS8 for the Phase 0/1 results
writeup once a benchmark sweep has been run.

## Repository layout

```
04-flashlite/
  pyproject.toml, setup.py     package metadata + CUDA extension build (ADR 0005)
  src/flashlite/
    reference/                 V0: trusted PyTorch reference + deterministic RNG
    compare.py, stats.py,      shared correctness/benchmark infrastructure
    timing.py, env_capture.py    (mirrors KernelForge's src/common/, ported to Python)
    bench_schema.py            benchmark result schema (Python side)
    ops.py                     variant dispatcher
    cuda_naive/                V1: naive CUDA kernel + pybind11 bindings
  tests/{math,correctness,edge_cases}/   pytest suite (see docs/architecture.md)
  benchmarks/{schema,configs,raw,methodology.md}
  docs/{attention-math.md,architecture.md,decisions/}
  scripts/                     build_ext.sh, test.sh, run_benchmarks.py, validate_results.py
```

## Setup and testing

All commands run **inside WSL2 Ubuntu**. From Windows:

```
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/guntr/Documents/passion-projects/projects/04-flashlite && python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && scripts/build_ext.sh && scripts/test.sh"
```

Or, from inside a WSL shell:

```bash
cd /mnt/c/Users/guntr/Documents/passion-projects/projects/04-flashlite
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
scripts/build_ext.sh     # pip install -e ".[dev]"; builds flashlite._cuda_naive
scripts/test.sh          # pytest -v; validates benchmarks/raw/*.jsonl against the schema
```

`scripts/build_ext.sh` sets `TORCH_CUDA_ARCH_LIST=8.9` (sm_89) in
`setup.py` so a fresh build does not depend on build-time GPU
auto-detection succeeding in every environment.

### Running just one test group

```bash
pytest tests/math -v            # Phase 0: reference correctness, RNG determinism, bench schema
pytest tests/correctness -v     # Phase 1: naive CUDA kernel vs reference (requires CUDA + built extension)
pytest tests/edge_cases -v      # unsupported-shape rejection paths (both V0 and V1)
pytest -k "causal" -v           # any test with "causal" in its name
```

Tests marked `gpu` are skipped automatically (with a specific reason) when
no CUDA device is present, or when CUDA is present but the extension has
not been built yet -- see `tests/conftest.py`.

### Reproducing the benchmarks

```bash
python scripts/run_benchmarks.py     # short Phase 0/1 sweep -> benchmarks/raw/attention.jsonl
python scripts/validate_results.py   # validate every committed result against the schema
```

`benchmarks/configs/attention.json` is the single source of truth for
exactly which shapes/variants/warmup/reps/seed were used; every field of
every committed `BenchResult` is described in `benchmarks/methodology.md`.

## What's implemented

- **V0 (`flashlite.reference.attention`)**: `S = QK^T/sqrt(d); P = softmax(S); O = PV`,
  via `torch.matmul` + `torch.softmax`, with an optional causal mask.
  Independently cross-checked against a from-scratch Python-loop
  brute-force computation (no `torch.matmul`/`torch.softmax` at all) and
  against `torch.nn.functional.scaled_dot_product_attention` (ADR 0006).
- **V1 (`flashlite._cuda_naive`)**: the same math, hand-written as three
  CUDA kernels (`compute_scores_kernel`, `softmax_rows_kernel`,
  `weighted_sum_kernel`), fully materializing the `[B, H, S, S]`
  score/probability matrix in global memory between kernels -- deliberately
  the simplest correct implementation, not a tuned one (see
  `docs/attention-math.md` SS5 for what V1 does and does not optimize, and
  what later phases change).

Supported shape/dtype/layout contract for both variants: `[batch, heads,
seq_len, head_dim]` contiguous float32 tensors, `q`/`k`/`v` sharing one
shape (self-attention MVP, ADR 0002), any positive dimensions including
awkward non-tile-multiple `seq_len` (V1 is untiled), causal or non-causal
(ADR 0003). Unsupported input raises a specific, documented exception --
see `tests/edge_cases/test_shape_contract.py`.

## Later phases (not yet implemented)

Memory accounting (Phase 2), tiling (Phase 3), online softmax (Phase 4,
with its own from-first-principles derivation in `docs/online-softmax.md`,
written independently before that kernel is implemented), fused IO-aware
attention (Phase 5), kernel tuning (Phase 6, where D5's shared-memory vs.
register-pressure tradeoff is resolved empirically), framework comparison
(Phase 7, optional Triton V5), and portfolio hardening (Phase 8) -- see
`04-flashlite-spec.md` Part 3 for the full roadmap.
