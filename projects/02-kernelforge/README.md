# KernelForge

CUDA kernel optimization laboratory. Full product spec: `02-kernelforge-spec.md`
(copied into this repo root). This README covers **Phase 0 (Harness &
correctness infra)** and **Phase 1 (Memory access lab)** — the only phases
implemented so far.

Target hardware: **NVIDIA GeForce RTX 4090, sm_89** (ADR 0001) — the only
GPU this repo is built and tested against. Build environment: **WSL2
Ubuntu on Windows 11**, CUDA toolkit 12.6, g++ 13.3, CMake 3.28, Ninja.

## Status

| Phase | Deliverables | Exit criterion | Status |
|---|---|---|---|
| 0 | CMake/CUDA project, capability check, device-info, error macros, deterministic reference/test utilities, benchmark schema | Vector add validates; benchmark output machine-readable | **Met** |
| 1 | Vector/SAXPY, transpose, coalescing/stride microbenchmarks | Repo empirically demonstrates contiguous vs. pathological access | **Met** |

See `docs/decisions/` for every architectural decision (D1-D8 from the
spec, plus two implementation choices), each with rationale and
consequences. See `benchmarks/methodology.md` for the full, numbers-backed
Phase 1 results writeup.

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
  apps/                              device-info, bench_vector_add, bench_saxpy,
                                     bench_transpose, bench_stride
  tests/                             kf_tests (45 correctness tests, no GPU-less mode)
  benchmarks/                        methodology.md, schema/, configs/, raw/, plots/
  docs/                              gpu-model.md, optimization-method.md, decisions/
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
build/tests/kf_tests                       # all 45 tests
build/tests/kf_tests --filter=Transpose    # substring filter on "Suite.Name"
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

This builds (if needed), runs all four `bench_*` binaries across every
`(variant, size)` point in `benchmarks/configs/*.json`, appends results to
`benchmarks/raw/*.jsonl`, and validates every record against
`benchmarks/schema/bench_result.schema.json`. A single family can be run
with e.g. `scripts/run_bench_transpose.sh`, or a single point can be run
directly: `build/apps/bench_transpose --variant v2_tiled --n 4096 --warmup 10 --reps 30`.

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

## Architectural decisions

All spec open decisions (D1-D8) plus two implementation choices are
recorded in `docs/decisions/`:

- 0001 — Target GPU architecture (sm_89, RTX 4090)
- 0002 — Minimum supported compute capability (8.9)
- 0003 — FP32-only MVP
- 0004 — Architecture-safe warp primitives (`_sync` family, full masks) — not yet exercised, recorded ahead of Phase 2
- 0005 — Benchmark noise control (clocks NOT locked under WSL2 — disclosed, not hidden)
- 0006 — SASS analysis depth (PTX default, SASS where it explains a result) — not yet exercised
- 0007 — CUB comparisons (deferred/stretch)
- 0008 — Templating by dtype (no, FP32 only)
- 0009 — Lightweight in-repo test framework instead of GoogleTest
- 0010 — Hand-rolled JSON for the benchmark result schema

## Known limitations (disclosed, not silently worked around)

- **GPU clocks are not locked.** `nvidia-smi -lgc ...` returns a
  permission error under this WSL2 guest. Every benchmark result records
  `locked_clocks: false` plus an observed (unlocked) clock snapshot — see
  ADR 0005.
- **Nsight Compute (`ncu`) is not installed** on this machine as of
  2026-08-17. Nsight Systems and Compute Sanitizer are present and
  working. This has no effect on Phases 0-1 (no profiler evidence is
  claimed here); it will need to be resolved before Phase 6.
- **Transpose ladder is not complete.** Only V1 (naive) and V2 (tiled,
  coalesced read+write) are implemented. V2's shared-memory tile is
  deliberately *unpadded*, so it carries a known (documented, not yet
  measured with `ncu`) shared-memory bank conflict on the transposed read
  — the V4 fix (padding) from the spec's ladder is explicit future work,
  not silently included or silently omitted (see
  `src/kernels/transpose/transpose_tiled.cuh`).

## What's next

Phase 2 (reduction & scan) is the next unimplemented phase per the
roadmap in `02-kernelforge-spec.md` Part 3 — not started.
