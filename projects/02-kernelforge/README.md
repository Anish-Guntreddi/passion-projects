# KernelForge

CUDA kernel optimization laboratory. Full product spec: `02-kernelforge-spec.md`
(copied into this repo root). This README covers **Phase 0 (Harness &
correctness infra)**, **Phase 1 (Memory access lab)**, **Phase 2
(Reduction & scan)**, and **Phase 3 (Atomics/histogram lab)** — the phases
implemented so far.

Target hardware: **NVIDIA GeForce RTX 4090, sm_89** (ADR 0001) — the only
GPU this repo is built and tested against. Build environment: **WSL2
Ubuntu on Windows 11**, CUDA toolkit 12.6, g++ 13.3, CMake 3.28, Ninja.

## Status

| Phase | Deliverables | Exit criterion | Status |
|---|---|---|---|
| 0 | CMake/CUDA project, capability check, device-info, error macros, deterministic reference/test utilities, benchmark schema | Vector add validates; benchmark output machine-readable | **Met** |
| 1 | Vector/SAXPY, transpose, coalescing/stride microbenchmarks | Repo empirically demonstrates contiguous vs. pathological access | **Met** |
| 2 | Reduction ladder (5 versions), 2 scan strategies, correctness across awkward sizes | ≥4 reduction versions and 2 scan strategies benchmarked | **Met** |
| 3 | Histogram/atomics lab: contention baseline, privatization, coarsening | Low- vs high-contention workloads documented with measured results | **Met** |

See `docs/decisions/` for every architectural decision (D1-D8 from the
spec, plus implementation choices — 11 ADRs total). See
`benchmarks/methodology.md` for the full, numbers-backed results writeup
(§8 Phase 1, §9 Phase 2, §10 Phase 3), and each kernel family's own
`README.md` (`src/kernels/{reduction,scan,histogram}/README.md`) for the
ladder table and every rung's written hypothesis.

## Repository layout

```
02-kernelforge/
  CMakeLists.txt, cmake/            CMake + CUDA build config
  src/common/                       error macros, timers, device query, RNG,
                                     CPU references, comparison, stats,
                                     benchmark schema, test framework
  src/kernels/vector/                vector_add, saxpy
  src/kernels/transpose/             transpose_naive (V1), transpose_tiled (V2)
  src/kernels/microbench/            stride_copy (dedicated coalescing microbenchmark)
  src/kernels/reduction/             5-rung reduction ladder + README (ladder hypotheses)
  src/kernels/scan/                  2 scan strategies + shared multi-block driver + README
  src/kernels/histogram/             3-rung atomics lab (baseline/privatized/coarsened) + README
  apps/                              device-info, bench_vector_add, bench_saxpy,
                                     bench_transpose, bench_stride, bench_reduction,
                                     bench_scan, bench_histogram
  tests/                             kf_tests (101 correctness tests, no GPU-less mode)
  benchmarks/                        methodology.md, schema/, configs/, raw/, plots/
  docs/                              gpu-model.md, optimization-method.md, decisions/ (11 ADRs)
  scripts/                           configure/build/test/bench-* .sh + Python drivers
```

## Building and testing

All commands run **inside WSL2 Ubuntu**. From Windows:

```
wsl -d Ubuntu -- bash -lc "/mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge/scripts/build.sh"
wsl -d Ubuntu -- bash -lc "/mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge/scripts/test.sh"
```

Or, from inside a WSL shell:

```bash
cd /mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge
scripts/configure.sh   # cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ...
scripts/build.sh       # cmake --build build
scripts/test.sh         # build (if needed) + run kf_tests + validate benchmarks/raw/*.jsonl against schema
```

`scripts/configure.sh` pins `nvcc` by explicit path
(`/usr/local/cuda/bin/nvcc`) rather than relying on `$PATH`, and defaults
`CMAKE_CUDA_ARCHITECTURES=89` (override with `KF_CUDA_ARCH=<n>` env var).

### Device info

```bash
scripts/run_device_info.sh          # human-readable
scripts/run_device_info.sh --json   # machine-readable
```

### Correctness tests

```bash
build/tests/kf_tests                       # all 101 tests
build/tests/kf_tests --filter=Transpose    # substring filter on "Suite.Name"
build/tests/kf_tests --filter=Reduction    # Phase 2 reduction ladder only
build/tests/kf_tests --filter=Scan         # Phase 2 scan strategies only
build/tests/kf_tests --filter=Histogram    # Phase 3 atomics lab only
```

Or via CTest: `cd build && ctest --output-on-failure`.

### Compute Sanitizer

```bash
scripts/run_sanitizer.sh memcheck    # or racecheck / initcheck / synccheck
```

Verified clean on this repo's full test suite (2026-08-17): `========= ERROR SUMMARY: 0 errors`.

### Reproducing the benchmarks

```bash
scripts/run_all_benchmarks.sh
```

This builds (if needed), runs all seven `bench_*` binaries across every
`(variant, size)` point in `benchmarks/configs/*.json`, appends results to
`benchmarks/raw/*.jsonl`, and validates every record against
`benchmarks/schema/bench_result.schema.json`. A single family can be run
with e.g. `scripts/run_bench_transpose.sh`, `scripts/run_bench_reduction.sh`,
`scripts/run_bench_scan.sh`, or `scripts/run_bench_histogram.sh` (the last
runs BOTH the uniform and skewed contention sweeps — Phase 3's exit
criterion). A single point can be run directly, e.g.:

```bash
build/apps/bench_transpose  --variant v2_tiled                  --n 4096    --warmup 10 --reps 30
build/apps/bench_reduction  --variant v4_vectorized_coarsened   --n 1048576 --warmup 10 --reps 30
build/apps/bench_scan       --variant hillis_steele --block-size 1024 --n 65536 --warmup 10 --reps 30
build/apps/bench_histogram  --variant v3_privatized_coarsened --contention skewed --num-bins 256 --n 2097152 --warmup 10 --reps 30
```

Every benchmark binary runs exactly one correctness check against the CPU
reference before any timing loop, and refuses to emit a result if it
fails (FR6). `scripts/bench_driver.py` additionally refuses to *write* any
result with `correctness_passed=false`.

## Phase 1 headline result

The stride/coalescing microbenchmark (`src/kernels/microbench/stride_copy.cuh`,
`benchmarks/raw/stride_copy.jsonl`) moves exactly the same number of useful
bytes at every stride — only the address distance between consecutive
threads' accesses changes:

| stride | effective bandwidth | vs. stride=1 |
|---:|---:|---:|
| 1   | 1376.1 GB/s | 1.00x |
| 8   | 327.7 GB/s  | 0.24x |
| 16  | 93.1 GB/s   | 0.068x |
| 128 | 39.8 GB/s   | 0.029x |

The same mechanism, inside a real kernel: `transpose_naive` (coalesced
read, pathological write) vs. `transpose_tiled` (shared-memory-staged,
both coalesced), at 8192x8192 (`benchmarks/raw/transpose.jsonl`):
**272 GB/s vs. 709 GB/s — a 2.6x speedup from changing exactly one
variable**, with `v2_tiled` reaching ~70% of the device's 1008.1 GB/s
theoretical peak bandwidth versus `v1_naive`'s ~27%. Full numbers, every
raw repetition, and the reasoning for each are in
`benchmarks/methodology.md` §8.

## Phase 2 headline result

The 5-rung reduction ladder (`src/kernels/reduction/` — V1-V4's numbers are
in `benchmarks/raw/reduction.jsonl`; V0, the pure-global-atomics baseline,
was added after that data was collected and awaits its own benchmark pass,
see that README's "Ladder history" note) is monotonically faster rung over
rung at the largest fully-HBM-bound size (67,108,864 elements): naive
interleaved addressing → sequential addressing is **1.51x**, →
warp-shuffle finalization is a further **1.10x**, → vectorized/coarsened
loads is a further **1.31x**, for a **2.18x V1→V4 speedup overall (430 →
936 GB/s, ~93% of theoretical peak)** — one variable changed per rung, all
five numerically agree with each other and with the CPU reference at
every tested size. The two scan strategies (`src/kernels/scan/`,
`benchmarks/raw/scan.jsonl`) show Hillis-Steele beating Blelloch by
1.25x-2.16x across every size up to the 2-level ceiling — the opposite of
what pure work-efficiency would predict, and reported as measured rather
than adjusted to fit the initial guess. Full numbers and interpretation:
`benchmarks/methodology.md` §9.

## Phase 3 headline result

The histogram/atomics lab (`src/kernels/histogram/`,
`benchmarks/raw/histogram_{uniform,skewed}.jsonl`) is Phase 3's exit
criterion made concrete: at 33,554,432 elements / 256 bins, the
global-atomic baseline is **1.93x slower under a high-contention (skewed)
input than a low-contention (uniform) one** (17.5 → 9.1 GB/s) — the
*only* variant whose throughput depends on the input's statistical shape.
Shared-memory privatization (V2) all but erases that dependence (111.8 →
162.4 GB/s, *faster* under skew, the opposite of the naive prediction —
reported as measured, see `benchmarks/methodology.md` §10 for the
plausible-but-unconfirmed mechanism); adding thread coarsening (V3)
removes the contention penalty outright — at the largest size the two
distributions sit within 4% of each other (794.4 vs 762.0 GB/s, uniform
ahead) — while delivering a **45x (uniform) to 84x (skewed) speedup over
the V1 baseline**. Full sweep, every size: `benchmarks/methodology.md` §10.

## Architectural decisions

All spec open decisions (D1-D8) plus implementation choices are recorded
in `docs/decisions/` (11 ADRs):

- 0001 — Target GPU architecture (sm_89, RTX 4090)
- 0002 — Minimum supported compute capability (8.9)
- 0003 — FP32-only MVP
- 0004 — Architecture-safe warp primitives (`_sync` family, full masks) — first exercised in Phase 2's `reduce_warp_shuffle`
- 0005 — Benchmark noise control (clocks NOT locked under WSL2 — disclosed, not hidden)
- 0006 — SASS analysis depth (PTX default, SASS where it explains a result) — not yet exercised
- 0007 — CUB comparisons (deferred/stretch)
- 0008 — Templating by dtype (no, FP32 only)
- 0009 — Lightweight in-repo test framework instead of GoogleTest
- 0010 — Hand-rolled JSON for the benchmark result schema
- 0011 — BenchResult schema extension for Phase 2/3 (`reduction`/`scan`/`histogram` families, `num_bins`/`contention_profile` fields)

## Known limitations (disclosed, not silently worked around)

- **GPU clocks are not locked.** `nvidia-smi -lgc ...` returns a
  permission error under this WSL2 guest. Every benchmark result records
  `locked_clocks: false` plus an observed (unlocked) clock snapshot — see
  ADR 0005.
- **Nsight Compute (`ncu`) is not installed** on this machine as of
  2026-08-17. Nsight Systems and Compute Sanitizer are present and
  working. This has no effect on Phases 0-3 (no profiler evidence is
  claimed here); it will need to be resolved before Phase 6.
- **Transpose ladder is not complete.** Only V1 (naive) and V2 (tiled,
  coalesced read+write) are implemented. V2's shared-memory tile is
  deliberately *unpadded*, so it carries a known (documented, not yet
  measured with `ncu`) shared-memory bank conflict on the transposed read
  — the V4 fix (padding) from the spec's ladder is explicit future work,
  not silently included or silently omitted (see
  `src/kernels/transpose/transpose_tiled.cuh`).
- **Scan supports n up to `block_size^2` elements only** (1,048,576 at
  `block_size=1024`). A 3rd scan level would be required beyond that —
  explicit future work; `scan_multilevel_launch` throws
  `std::runtime_error` rather than silently truncating or misbehaving (see
  `src/kernels/scan/scan_common.cuh`).
- **Histogram's privatized shared-memory histogram requires
  `num_bins * 4 <= 49,152` bytes** (this GPU's default per-block shared
  memory). `histogram_common.cuh::validate_num_bins_fits_shared_mem` fails
  loudly if exceeded rather than letting the launch fail late.

## What's next

Phase 4 (GEMM: naive → tiled → register-tiled ladder, cuBLAS ceiling
comparison) is the next unimplemented phase per the roadmap in
`02-kernelforge-spec.md` Part 3 — not started.
