# KernelForge — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** KernelForge — CUDA Kernel Optimization Laboratory
**Portfolio position:** 02 of 09 · Track B (GPU performance) · prerequisite for FlashLite
**Source of truth:** "02 - KernelForge - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
KernelForge is a CUDA performance-engineering laboratory. The goal is not a pile of toy kernels; it is a **repeatable optimization process**: establish a correct baseline, form a hardware hypothesis, profile, change one variable, re-profile, and explain the result using GPU architecture.

### 1.2 Core product contract
Every kernel family follows the same pipeline:
**Reference implementation → naïve CUDA → measured bottleneck → optimized variant(s) → correctness verification → benchmark report → profiler evidence → short postmortem.**

### 1.3 Functional requirements (MVP)
- **FR1** Kernel families implemented: vector/SAXPY baseline, reduction, prefix scan, matrix transpose, histogram, tiled GEMM, softmax, RMSNorm or LayerNorm.
- **FR2** Required optimization ladders (each version is a separate, preserved variant):
  - *Reduction:* V0 CPU ref → V1 naïve global-memory → V2 shared-memory → V3 reduced divergence/sequential addressing → V4 warp-level primitives → V5 vectorized/coarsened loads.
  - *Transpose:* V0 ref → V1 naïve → V2 coalesced read/write → V3 shared-memory tile → V4 bank-conflict padding.
  - *GEMM:* V0 ref → V1 one thread/output → V2 coalesced cleanup → V3 shared-memory tiling → V4 register tiling/coarsening → V5 optional vectorized/double-buffered. cuBLAS is a **ceiling/reference, never a target to beat**.
  - *Softmax/norm:* simple CUDA first, then reduction strategy, memory traffic, bounded fusion, numerical stability.
- **FR3** Common infrastructure: CUDA error macros, timers (CUDA events), device-info command, deterministic reference/test utilities, machine-readable benchmark result schema.
- **FR4** Benchmark contract — every result records: GPU model + compute capability; CUDA/toolkit/compiler versions; clock/power caveats; input dimensions; block/grid config; warmup count; measured repetitions; median + distribution (never a single timing); effective bandwidth or FLOP/s where meaningful; correctness tolerance; compiler flags. Kernel time and transfer time are never mixed unless explicitly measuring end-to-end (report both when useful).
- **FR5** Profiling contract — Nsight Systems for timeline/system questions, Nsight Compute for kernel questions. Every major optimization report follows: **Hypothesis → evidence before → change → evidence after → measured outcome → interpretation.** Collect only metrics relevant to the hypothesis.
- **FR6** ≥3 kernels with complete profiler-backed case studies, including PTX (optionally SASS) notes for selected kernels.
- **FR7** Analysis pipeline: Python (pandas/matplotlib) converts committed raw results into plots. CUDA implementation stays C++.

### 1.4 Non-goals
Reproducing cuBLAS/cuDNN/CUB internally; claiming to beat vendor libraries; implementing FlashAttention here (**FlashLite owns attention**); supporting every NVIDIA architecture; production serving; multi-node CUDA.

### 1.5 Deliverable artifacts (website/resume)
Optimization ladder per kernel; before/after throughput graphs; roofline or bandwidth-efficiency plots where appropriate; Nsight screenshots or derived metric tables; a GPU architecture diagram explaining one representative bottleneck; PTX/SASS notes. Website packaging: each highlighted kernel gets a card — *Problem → baseline bottleneck → optimization → measured improvement → profiler evidence*; pick the 2–3 strongest stories for the website, keep the rest on GitHub. Resume narrative filled only from committed evidence: *"Built a CUDA optimization laboratory spanning reductions, scans, transpose, histogram, GEMM, softmax and normalization kernels; developed optimization ladders from naïve implementations through memory-, warp- and tiling-aware variants and validated improvements with Nsight profiling."* **No speedup number is prewritten.**

### 1.6 Test requirements
Randomized correctness vs CPU/reference; edge sizes (0/1 where legal, non-power-of-two, non-block-multiple, small/large); numerical tolerances by dtype; CUDA error checks after every launch; Compute Sanitizer workflow; deterministic seeds; correctness check runs before any timing.

### 1.7 Acceptance criteria (MVP complete when)
All required families have correct baselines; ≥4 families have multi-stage ladders; benchmarks reproducible from scripts; device/toolchain metadata captured automatically; ≥3 profiler-backed case studies; every claimed speedup traceable to raw committed results; README explains what **hardware mechanism** changed.

### 1.8 Open decisions (recommended defaults)
- **D1** Target GPU architecture available to developer → *human decision; record in device-info output.*
- **D2** Minimum supported compute capability → *default: whatever D1's card supports; guard newer features behind capability checks.*
- **D3** FP32-only MVP → *default: yes.*
- **D4** Architecture-safe warp primitives → *default: __shfl_sync family with full masks.*
- **D5** Benchmark noise control → *default: locked clocks documented if possible; ≥20 reps, median + IQR.*
- **D6** SASS analysis depth → *default: PTX for 3 case studies; SASS only where it explains a result.*
- **D7** CUB comparisons → *default: stretch.*
- **D8** Templating by dtype → *default: no for MVP (FP32 only per D3).*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Core | C++20 + CUDA C++ | Brief requirement |
| Build | CMake + Ninja | Brief requirement |
| Testing | GoogleTest (or lightweight deterministic C++ framework) | Brief requirement |
| Benchmarking | Custom harness on CUDA events; optional Google Benchmark for CPU-side | Brief requirement |
| Profiling | Nsight Systems, Nsight Compute, Compute Sanitizer | Brief requirement |
| Analysis | Python + pandas/matplotlib (aggregation/plots only) | Brief requirement |
| CI | Build/test CPU-host components on standard runner; optional GPU CI when practical | Brief requirement |

### Repository structure
```
kernelforge/
  CMakeLists.txt
  cmake/
  src/common/                 # error macros, timers, device info, bench schema, references
  src/kernels/{vector,reduction,scan,transpose,histogram,gemm,softmax,norm}/
  tests/
  benchmarks/{configs,raw,plots}/  benchmarks/methodology.md
  profiling/{nsight-systems,nsight-compute}/  profiling/metric-guide.md
  analysis/
  docs/gpu-model.md  docs/optimization-method.md  docs/decisions/
  scripts/
```
Every kernel exposes a common conceptual interface: input allocation/prep → launch wrapper → sync/error check → output validation → timing metadata.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Harness & correctness infra** | CMake project, capability check, device-info command, error macros, deterministic reference/test utilities, benchmark schema | Vector add validates; benchmark output machine-readable |
| **1 — Memory access lab** | Vector/SAXPY, transpose, dedicated coalescing/stride microbenchmarks | Repo empirically demonstrates contiguous vs pathological access |
| **2 — Reduction & scan** | Full ladders + correctness tests across awkward sizes | ≥4 reduction versions and 2 scan strategies benchmarked |
| **3 — Atomics/histogram** | Contention baseline, privatization, coarsening experiments | Low- vs high-contention workloads documented |
| **4 — GEMM** | Naïve → tiled → register-tiled ladder; cuBLAS ceiling comparison where available | Report explains memory reuse and resource tradeoffs |
| **5 — AI primitives** | Softmax + RMSNorm/LayerNorm with references and optimization experiments | Benchmarks across representative tensor shapes |
| **6 — Profiling & low-level analysis** | 3 representative kernels: Nsight Systems/Compute reports, PTX (optional SASS) inspection | 3 complete optimization case studies |
| **7 — Portfolio release** | Plots, README, methodology, architecture notes, reproducibility scripts | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
Tensor Core GEMM; CUTLASS comparison; CUDA Graph launch-overhead experiment; async copy/double buffering; multi-GPU bandwidth microbenchmarks; persistent kernels; custom allocator for benchmark reuse; Hopper/Blackwell comparative runs if hardware available.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Correctness before performance.
2. Never copy an optimized kernel wholesale from external source code.
3. Never claim speedup without benchmark evidence.
4. One optimization variable per experiment where possible.
5. Keep reference and optimized variants side-by-side for learning.
6. Preserve readable CUDA suitable for interview discussion.
7. Use profiling evidence to justify changes.
8. Fail loudly on unsupported hardware features (capability checks).
9. Separate implementation commits from benchmark-result commits.
10. Require a written optimization hypothesis before any performance change.

### Kickoff prompt
> Read `02-kernelforge-spec.md` in full. Produce an engineering plan that: bootstraps CMake/CUDA/test infrastructure first (Phase 0); implements one kernel family at a time in Part 3 order; creates correctness tests before optimization variants; defines the benchmark result schema and `benchmarks/methodology.md` before collecting any headline numbers; schedules a profiling checkpoint after each family baseline; separates implementation commits from benchmark-result commits; requires a written hypothesis before each optimization variant; includes acceptance criteria and exact reproducibility commands per task; and puts all architecture-specific code behind capability checks. Resolve decisions D1–D8 with me before Phase 0 where marked human. Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (after each ladder)  →  /ship
```
Skip `/qa` (no runnable UI) and `/cso` (no sensitive surface). Use `/benchmark` heavily — it is the core of this project.
