# Profiling Metric Guide

Phase 6 ("Profiling & low-level analysis") deliverable. See
`docs/decisions/0014-phase6-profiling-evidence-strategy.md` for the full
investigation of what's usable in this environment and why; this file is
the "which evidence source answers which kind of question" reference the
3 case studies (`profiling/case-studies/`) point back to.

## What's here

- `occupancy/` — `occupancy_report` JSON records (schema:
  `profiling/schema/occupancy_report.schema.json`), one per
  `(kernel_family, variant, launch shape)` queried. Theoretical occupancy
  from the CUDA Runtime's own occupancy API, computed from the actual
  compiled kernel's register/shared-memory footprint.
- `nsight-systems/` — `.nsys-rep` captures + a `nsys stats` text summary
  per capture. Contains real, populated **host-side CUDA API** data
  (`cudaLaunchKernel`/`cudaMemcpy`/`cudaEventSynchronize` call counts and
  durations); **device-side GPU kernel/memory trace sections are
  genuinely empty** in every capture here (`nsys stats` reports `SKIPPED:
  ... does not contain CUDA kernel data` — this is not this repo's bug,
  see ADR 0014, cross-checked against FlashLite's independently-captured
  `.nsys-rep` files showing the identical result).
- `nsight-compute/` — no `.ncu-rep` files; `README.md` there documents
  exactly why (`ERR_NVGPUCTRPERM`, a WSL2-host-driver permission
  restriction, reproduced with the exact command and error).
- `ptx-sass/` — `cuobjdump --dump-ptx` / `--dump-sass` output for the
  benchmark binaries covering the 3 case-study kernel families
  (transpose, reduction, gemm). Static disassembly, needs no GPU access.
- `case-studies/` — the 3 required profiler-backed writeups (FR6),
  combining the sources above with already-committed `benchmarks/raw/`
  wall-clock evidence.

## Which evidence source answers which kind of question

| Question | Source | Where |
|---|---|---|
| "How many threads/blocks can be co-resident on one SM for this exact compiled kernel + launch config?" (theoretical occupancy) | CUDA Runtime occupancy API | `profiling/occupancy/*.jsonl` |
| "How many registers does this kernel actually use? How much shared memory?" | `cudaFuncGetAttributes` (embedded in the occupancy record) | `profiling/occupancy/*.jsonl` — `registers_per_thread`, `static_shared_bytes_per_block` |
| "Given this problem shape's grid, what fraction of the WHOLE device's block-residency capacity is used?" (grid-level utilization, distinct from per-SM occupancy) | `occupancy_report`'s `--m`/`--n` grid math | `profiling/occupancy/gemm.jsonl` — `grid_utilization_fraction` |
| "Does this kernel actually compile to the memory-access pattern the hypothesis describes (coalesced vs. strided; bank-conflicted vs. not)?" | SASS disassembly (`LDG`/`STG`/`LDS`/`STS` addressing) | `profiling/ptx-sass/*.sass.txt` |
| "How many synchronization barriers / warp-shuffle instructions does this kernel issue?" | SASS instruction counts (`BAR.SYNC`, `SHFL.DOWN`) | `profiling/ptx-sass/*.sass.txt` |
| "How many times was this kernel actually launched? What did the host spend time on around it (allocation, transfer, sync)?" | `nsys`'s host-side CUDA API trace | `profiling/nsight-systems/*.stats.txt` (`cuda_api_sum` table) |
| "How long did the kernel itself take, across many repetitions, with a real distribution?" | This repo's own `cudaEvent`-based timing (unaffected by the WSL2 restriction — the app times itself) | `benchmarks/raw/*.jsonl` (already committed in Phases 1-5) |
| "What would a hardware performance counter (achieved occupancy, warp execution efficiency, DRAM/L2 hit rate) show?" | **Not available in this environment** (`ERR_NVGPUCTRPERM`) | `profiling/nsight-compute/README.md` documents the attempt; no case study in this repo claims a hardware-counter number |

## Verified tool availability (2026-08-17, WSL2, re-checked at Phase 6 start)

```
$ nsys --version
NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0
$ /usr/local/cuda/bin/ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler, Version 2024.3.2.0 -- installed, but ERR_NVGPUCTRPERM on actual use (ADR 0014)
$ /usr/local/cuda/bin/cuobjdump --version | head -2
Cuobjdump: NVIDIA (R) Cuda Object Dump Utility, Version 12.6.85
$ /usr/local/cuda/bin/nvdisasm --version | head -2
nvdisasm: NVIDIA (R) CUDA disassembler, Release 12.6, V12.6.85
$ /usr/local/cuda/bin/compute-sanitizer --version
NVIDIA (R) Compute Sanitizer, Version 2024.3.0.0 (public-release)
```
