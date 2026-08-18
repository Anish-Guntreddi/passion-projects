# FlashLite

From naive attention to IO-aware fused attention. Full product spec:
`04-flashlite-spec.md` (repo root of `passion-projects`, one level above
this project). This README covers **Phase 0 (Math/reference harness)**,
**Phase 1 (Naive custom attention)**, **Phase 2 (Memory accounting)**,
**Phase 3 (Tiling)**, **Phase 4 (Online softmax)**, and **Phase 5 (Fused
IO-aware attention)** -- the phases implemented so far.

Target hardware: **NVIDIA GeForce RTX 4090, sm_89** (ADR 0001, matching
KernelForge's own target-architecture ADR) -- the only GPU this repo is
built and tested against. Build environment: **WSL2 Ubuntu on Windows 11**,
CUDA toolkit 12.6, Python 3.12, PyTorch 2.13.0+cu126.

## Status

| Phase | Deliverables | Exit criterion | Status |
|---|---|---|---|
| 0 | Reference attention, configurable causal masking, deterministic tensors, correctness comparator, benchmark result schema | Trusted PyTorch outputs exist across representative shapes | **Met** |
| 1 | Score computation, softmax, value aggregation as straightforward CUDA kernels | Custom path matches reference within documented tolerance | **Met** |
| 2 | Theoretical bytes/FLOPs calculation + measured profiler baseline before optimizing | Baseline bottleneck hypothesis documented | **Met** |
| 3 | Q/K/V tiled with shared/on-chip memory; conceptually simple softmax retained | Tiled implementation correct and benchmarked | **Met** |
| 4 | Running max + normalization accumulator implemented/tested independently, then integrated | Tests cover extreme score values and multiple tile boundaries | **Met** |
| 5 | Full attention-matrix materialization removed; output accumulated tile-by-tile | Peak-memory scales as designed; output matches reference | **Met** |

See `docs/decisions/` for the spec's open decisions D1-D4, D6, D7 (D5 is
explicitly deferred to Phase 6, resolved empirically) plus the
implementation ADRs: 0007 (V2 tile-size provisional default), 0008
(BenchResult schema `tile_size` field), 0009 (V3 online-softmax kernel
design), 0010 (BenchResult schema `peak_memory_bytes` field), and 0011 (V4
fused-kernel design). See `docs/attention-math.md` for the V0/V1 math
derivation and materialized-form cost analysis, `docs/online-softmax.md`
for the from-first-principles online-softmax derivation (spec SS1.4, Phase
4), `docs/io-analysis.md` for the Phase 2/3 memory-accounting hypothesis
and measured results (plus its Phase 4/5 addendum, SS9), and
`docs/architecture.md` for the data-flow/package-layout overview. See
`benchmarks/methodology.md` SS8/SS9/SS10 for the Phase 0/1, Phase 2/3, and
Phase 4/5 results writeups.

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
    online_softmax.py          Phase 4: standalone online-softmax "unit" (docs/online-softmax.md),
                                 no CUDA dependency, tested independently before any kernel
    cuda_common/                shared CUDA/C++ infrastructure (Phase 3): error-check
                                 macros + the shared shape/dtype/layout validator
    cuda_naive/                V1: naive CUDA kernel + pybind11 bindings
    cuda_tiled/                V2 (Phase 3): shared-memory-tiled CUDA kernel + bindings
    cuda_online_softmax/       V3 (Phase 4): kernel 1/3 = V2's, kernel 2 = combined-pass online softmax
    cuda_fused/                V4 (Phase 5): single fused kernel, no [B,H,S,S] buffer at all
  tests/{math,correctness,edge_cases}/   pytest suite (see docs/architecture.md)
  benchmarks/{schema,configs,raw,methodology.md}
  profiling/nsight-systems/     Phase 2: nsys captures + CUDA-API CSV extracts
  docs/{attention-math.md,online-softmax.md,io-analysis.md,architecture.md,decisions/}
  scripts/                     build_ext.sh, test.sh, run_benchmarks.py,
                                validate_results.py, profile_kernels.py
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
scripts/build_ext.sh     # pip install -e ".[dev]"; builds all four CUDA extensions
                          # (_cuda_naive, _cuda_tiled, _cuda_online_softmax, _cuda_fused)
scripts/test.sh          # pytest -v; validates benchmarks/raw/*.jsonl against the schema
```

`scripts/build_ext.sh` sets `TORCH_CUDA_ARCH_LIST=8.9` (sm_89) in
`setup.py` so a fresh build does not depend on build-time GPU
auto-detection succeeding in every environment.

### Running just one test group

```bash
pytest tests/math -v            # Phase 0/4: reference correctness, RNG determinism, bench schema,
                                 # + tests/math/test_online_softmax.py (CPU-only, no CUDA required)
pytest tests/correctness -v     # Phase 1/3/4/5: naive/tiled/online-softmax/fused CUDA kernels vs
                                 # reference (requires CUDA + built extensions)
pytest tests/edge_cases -v      # unsupported-shape rejection paths (V0-V4)
pytest -k "causal" -v           # any test with "causal" in its name
pytest -k "tiled" -v            # Phase 3 only: V2 correctness + edge cases
pytest -k "online_softmax" -v   # Phase 4 only: V3 correctness + edge cases (CUDA-integration half);
                                 # tests/math/test_online_softmax.py is the CPU-only half
pytest -k "fused" -v            # Phase 5 only: V4 correctness, edge cases, and peak-memory-scaling test
```

Tests marked `gpu` are skipped automatically (with a specific reason) when
no CUDA device is present, or when CUDA is present but one of the compiled
extensions has not been built yet -- see `tests/conftest.py`.

### Reproducing the benchmarks

```bash
python scripts/run_benchmarks.py                                                    # Phase 0/1 sweep
python scripts/run_benchmarks.py --config benchmarks/configs/memory_accounting.json # Phase 2 sweep
python scripts/run_benchmarks.py --config benchmarks/configs/tiled_comparison.json  # Phase 3 sweep
python scripts/run_benchmarks.py --config benchmarks/configs/phase4_5_comparison.json # Phase 4/5 sweep
                                                                                       # (adds peak_memory_bytes, ADR 0010)
python scripts/validate_results.py   # validate every committed result against the schema
```

All four append to `benchmarks/raw/attention.jsonl`. Each
`benchmarks/configs/*.json` file is the single source of truth for exactly
which shapes/variants/warmup/reps/seed were used; every field of every
committed `BenchResult` is described in `benchmarks/methodology.md`.
**Any of the above requires the GPU-benchmark lock if its numbers will be
cited in committed docs** -- correctness-only test runs do not.

Reproducing the Phase 2 Nsight Systems capture (`docs/io-analysis.md`
SS5.2; also requires the GPU-benchmark lock):

```bash
nsys profile -o profiling/nsight-systems/<name> -f true python scripts/profile_kernels.py \
    --variant naive --batch 1 --heads 8 --seq-len 4096 --head-dim 64 --iters 20 --warmup 5
nsys stats --report cuda_api_sum profiling/nsight-systems/<name>.nsys-rep
```

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
- **V2 (`flashlite._cuda_tiled`)**: the same math, with kernels 1 and 3
  (`compute_scores_tiled_kernel`, `weighted_sum_tiled_kernel`) rewritten to
  stage Q/K (resp. P/V) through `32x32` shared-memory tiles instead of
  reading straight from global memory per thread (ADR 0007: provisional
  `kAttnTileDim=32`, revisited empirically in Phase 6 per spec D5). Kernel
  2 (softmax) is byte-for-byte V1's `softmax_rows_kernel`, unchanged --
  "conceptually simple softmax retained" is this project's literal Phase 3
  scope. See `docs/io-analysis.md` for the theoretical prediction and
  measured speedup this produces (a `1.5-1.7x` end-to-end latency
  improvement over V1 at `seq_len>=512`, well below the tiled kernels'
  own `~15-25x` arithmetic-intensity improvement, because the untouched
  softmax kernel and shared per-call overhead remain a binding constraint).
- **V3 (`flashlite._cuda_online_softmax`)**: the same math, with kernel 2
  replaced by a single combined-pass online-softmax row normalization
  (running max `m` + running normalization sum `l`, updated per-thread then
  combined by a block-level tree reduction, `docs/online-softmax.md`)
  instead of V1/V2's two-pass max-then-sum reduction -- two global-memory
  passes over each row instead of three. Kernels 1 and 3
  (`compute_scores_tiled_kernel`, `weighted_sum_tiled_kernel`) are
  byte-for-byte V2's, unchanged (ADR 0009): the full `[B, H, S, S]`
  score/probability matrix is **still fully materialized** between kernels
  -- V3's one variable is the softmax *algorithm*, not memory layout. The
  standalone recurrence (`combine_online_stats`, `online_softmax_stats`,
  `online_softmax_attention_row`) is implemented and unit-tested in plain
  Python/PyTorch (`src/flashlite/online_softmax.py`,
  `tests/math/test_online_softmax.py`, CPU-only) **before** the CUDA
  kernel, per spec SS1.4's hard requirement. Measured effect: a modest
  (`~1-3%` at `seq_len>=512`) end-to-end speedup over V2, exactly as
  predicted (`benchmarks/methodology.md` SS10.2) -- kernel 2's traffic
  reduction is real but small relative to kernels 1/3's cost, which V3
  leaves untouched.
- **V4 (`flashlite._cuda_fused`)**: a single kernel, one thread per query
  row (`kFusedTileDim=32` rows per block), that streams K/V in
  `kFusedTileDim`-row tiles through shared memory and applies the identical
  online-softmax recurrence directly to freshly-computed `Q K^T` scores, one
  key column at a time, into a private per-thread `(m, l, acc[D])` state --
  **no `[B, H, S, S]` scores buffer is ever allocated** (ADR 0011,
  `attention_fused.cuh`). This is Phase 5's exit criterion, measured:
  peak allocated memory grows close to linearly with `seq_len` (`~1.4-1.7x`
  per doubling) versus the materializing variants' `~2.2-3.5x` trending
  toward the `4x` quadratic signature -- a `1.05x -> 10.12x` widening
  materialized/fused peak-memory ratio across `seq_len in [128, 4096]`
  (`benchmarks/methodology.md` SS10.1,
  `tests/correctness/test_fused_attention.py`). Honestly reported: V4 is
  currently **slower** in wall-clock latency than V2/V3 at every tested
  shape (`benchmarks/methodology.md` SS10.3) -- the accepted cost of
  trading V1-V3's `32x32=1024`-thread-per-block parallelism for
  one-thread-per-row, fixed-size per-thread register state, deferred to
  Phase 6 tuning (ADR 0011), not a correctness problem.

Supported shape/dtype/layout contract for all five variants: `[batch,
heads, seq_len, head_dim]` contiguous float32 tensors, `q`/`k`/`v` sharing
one shape (self-attention MVP, ADR 0002), any positive dimensions
including awkward non-tile-multiple `seq_len` -- V1 is untiled by
definition, and V2/V3's boundary-checked tile loads are correct for
non-tile-multiple shapes too, not just a special case -- causal or
non-causal (ADR 0003). Unsupported input raises a specific, documented
exception (shared between V1-V4 via `cuda_common/shape_validate.hpp`) --
see `tests/edge_cases/test_shape_contract.py`. V4 additionally requires
`head_dim <= 128` (`kMaxHeadDimFused`, ADR 0011; every head_dim this
repo's test matrix uses -- 32/64/128 -- is within that bound), enforced by
its own `TORCH_CHECK` in `cuda_fused/bindings.cpp`.

## Later phases (not yet implemented)

Kernel tuning (Phase 6, where D5's shared-memory vs. register-pressure
tile-size tradeoff is resolved empirically -- explicitly deferred by ADR
0007/0011 rather than guessed at during Phases 3/5), framework comparison
(Phase 7, vs. PyTorch SDPA + optional Triton V5), and portfolio hardening
(Phase 8) -- see `04-flashlite-spec.md` Part 3 for the full roadmap.
