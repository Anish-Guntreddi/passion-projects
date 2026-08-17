# FlashLite — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** FlashLite — From Naïve Attention to IO-Aware Fused Attention
**Portfolio position:** 04 of 09 · Track A/B convergence · after KernelForge · optional Helios/MiniPaged backend later
**Source of truth:** "04 - FlashLite - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
FlashLite demonstrates why attention performance is dominated by **memory movement**, not FLOP count alone. The repo progresses from an obviously correct PyTorch reference through naïve CUDA implementations toward a tiled, online-softmax attention kernel inspired by FlashAttention principles — kept small enough that the author can explain the complete forward kernel on a whiteboard.

### 1.2 Mathematical scope
Forward causal/non-causal scaled dot-product attention for one defined tensor layout, `[batch, heads, sequence, head_dim]`:
```
S = QK^T / sqrt(d);   P = softmax(S);   O = PV
```
The optimized path avoids materializing S/P by tiling Q/K/V with online softmax statistics. **Backward pass is explicitly stretch** unless the plan shows forward completes cleanly first.

### 1.3 Functional requirements — the variant ladder (each preserved as a separate implementation)
- **V0** PyTorch reference (standard tensor ops).
- **V1** Naïve custom kernel materializing scores/output intermediates.
- **V2** Tiled computation using shared/on-chip memory.
- **V3** Numerically stable online softmax across K/V tiles.
- **V4** Fused tiled attention avoiding full score/probability materialization.
- **V5** Optional Triton (or CUDA) counterpart to compare programming models.

The project succeeds **even if the custom kernel does not beat the platform's built-in attention** — the value is the algorithmic analysis and evidence.

### 1.4 Online-softmax documentation requirement (hard requirement)
`docs/online-softmax.md` must explicitly derive the running update (running maximum `m`, normalization accumulator `l`) and cover: why naïve exponentiation is unstable; why subtracting the row maximum is stable; how running maxima change across tiles; how prior partial outputs are rescaled when a new maximum appears; how the final normalized output is produced. **This may not be hidden behind a copied implementation.** The V3 online-softmax unit is implemented and tested independently before integration into attention.

### 1.5 Non-goals
Reproducing the production FlashAttention library; supporting every dtype/layout/mask; full Transformer serving; multi-GPU attention; claiming benchmark superiority over vendor/library kernels; starting with Hopper/Blackwell-specific instructions before a portable baseline works.

### 1.6 Deliverable artifacts (website/resume)
Diagram comparing materialized vs tiled/streaming attention; memory-complexity graphic; latency-vs-sequence-length plot; peak-memory-vs-sequence-length plot; selected Nsight memory-traffic comparison; numerical-error table vs PyTorch reference. Resume narrative filled from evidence: *"Implemented an IO-aware attention kernel from first principles, progressing from materialized attention to tiled online-softmax variants in CUDA/Triton; benchmarked latency, memory use and HBM traffic across sequence lengths and head dimensions."*

### 1.7 Test matrix
Shapes: sequence length tiny/moderate/large within hardware limits; head_dim 32/64/128 where supported; varied batch/head counts; causal and non-causal if both supported; awkward non-tile-multiple lengths.
Numerics: FP32 baseline; optional FP16/BF16 after FP32 correctness; extreme logits to stress softmax stability; random seeds + deterministic cases.
Correctness metrics: max absolute error, mean absolute error, relative tolerance appropriate to dtype. Unsupported shapes must **fail clearly** (documented supported-shape contract per variant).

### 1.8 Benchmark plan
Measure: kernel latency; end-to-end invocation latency; peak allocated memory; memory-traffic metrics where the profiler supports them; achieved throughput if meaningful; sequence-length scaling; head-dim sensitivity.
Comparison set: PyTorch reference, naïve, tiled, online/fused, platform optimized backend (SDPA).
Every result records GPU/toolkit/dtype/shape/warmup/repetitions.

### 1.9 Engineering quality bar
No giant monolithic kernel without staged variants; every variant has a correctness test; online-softmax math documented independently of code; optimizations justified by memory/compute reasoning; kernel launch/config decisions documented; benchmark plots generated from committed raw data.

### 1.10 Open decisions (recommended defaults)
- **D1** CUDA-first vs Triton-first → *default per brief: CUDA C++ primary; Triton as later comparison phase.*
- **D2** Supported tensor layout → *default: [B, H, S, D] contiguous only.*
- **D3** Causal-only vs causal+non-causal MVP → *default: both, causal via flag; if scope pressure, causal-only with ADR.*
- **D4** Dtype sequence → *default: FP32 first, then FP16/BF16.*
- **D5** Shared-memory vs register-pressure tile constraints on the available GPU → *resolved empirically in Phase 6.*
- **D6** CUDA-extension ↔ Python integration route → *default: pybind11/torch cpp_extension; ADR.*
- **D7** Baseline comparison backend → *default: torch.nn.functional.scaled_dot_product_attention.*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Core kernels | CUDA C++ (primary), Triton (later comparison) | D1 / brief's preferred portfolio path |
| Reference/testing | Python + PyTorch, pytest + kernel-level correctness harness | Brief requirement |
| Build | CMake / torch cpp_extension + pybind route | D6 |
| Profiling | Nsight Systems/Compute; Triton profiler / torch profiler as auxiliaries | Brief requirement |
| Analysis | Python/matplotlib from committed raw results | Brief requirement |
| CI | Lint + CPU-side tests; GPU tests local/optional runner | Practicality |

### Repository structure
```
flashlite/
  pyproject.toml
  CMakeLists.txt                # CUDA extension
  src/reference/
  src/cuda/{naive,tiled,online_softmax,fused}/
  src/triton/                   # optional later
  src/bindings/
  tests/{math,correctness,edge_cases}/
  benchmarks/{configs,raw,plots}/  benchmarks/methodology.md
  profiling/
  docs/attention-math.md  docs/online-softmax.md  docs/io-analysis.md  docs/architecture.md  docs/decisions/
```
Architecture: Python benchmark/test harness → reference attention → custom extension/kernel dispatcher (naïve | tiled | online-softmax | fused) → correctness comparator → timing/memory profiler → raw results/plots. Variants live side-by-side so Git history and the benchmark report show the optimization ladder.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Math/reference harness** | Reference attention, configurable causal masking, deterministic tensors, correctness comparator, benchmark result schema | Trusted PyTorch outputs exist across representative shapes |
| **1 — Naïve custom attention** | Score computation, softmax, value aggregation as straightforward kernels | Custom path matches reference within documented tolerance |
| **2 — Memory accounting** | Theoretical bytes/FLOPs calculation + measured profiler baseline **before optimizing** | Baseline bottleneck hypothesis documented |
| **3 — Tiling** | Q/K/V tiled with shared/on-chip memory; conceptually simple softmax retained | Tiled implementation correct and benchmarked |
| **4 — Online softmax** | Running max + normalization accumulator implemented/tested **independently**, then integrated | Tests cover extreme score values and multiple tile boundaries |
| **5 — Fused IO-aware attention** | Full attention-matrix materialization removed; output accumulated tile-by-tile | Peak-memory scales as designed; output matches reference |
| **6 — Kernel tuning** | Tile-size/block-shape sweep for supported shape family; profile throughput, occupancy, stalls | Tuned defaults based on benchmark evidence, not folklore |
| **7 — Framework comparison** | Compare vs PyTorch SDPA/platform backend; optional Triton implementation + authoring/runtime tradeoff writeup | Comparison report committed |
| **8 — Portfolio hardening** | Algorithm walkthrough, benchmark report, diagrams, README | Fresh-clone build/test/benchmark commands verified |

### Stretch goals (post-MVP only)
Backward pass; dropout; GQA/MQA layout; FP8; Hopper TMA/warp-specialization study; Blackwell experiment; variable-length/paged attention interface; integration into MiniPaged/Helios as an optional backend.

### Definition of Done
Naïve and fused/online variants correct; final optimized path does **not** materialize the full attention matrix; memory/latency scaling benchmarked; online softmax explained mathematically; profiler evidence supports the IO story; reproducible build/test/benchmark commands; website-ready comparison diagrams and plots.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Never copy the FlashAttention production implementation wholesale.
2. Derive algorithms from documented math and validate incrementally.
3. Keep the optimized implementation readable.
4. Correctness and stability precede speed.
5. Never suppress a numerical mismatch to improve benchmark appearance.
6. Never claim a speedup without raw reproducible results.

### Per-task attachments (brief mandate)
Each implementation task carries: the mathematical invariant; expected tensor shapes/layout; a correctness test; the supported/unsupported shape contract; a benchmark or profiler checkpoint; and the acceptance command.

### Kickoff prompt
> Read `04-flashlite-spec.md` in full. Produce an engineering plan in strict dependency order: reference harness → naïve kernel → memory analysis → tiled kernel → online-softmax unit → fused kernel → tuning → comparison → documentation. Attach to every implementation task its mathematical invariant, tensor shape/layout contract, correctness test, supported/unsupported shape contract, benchmark/profiler checkpoint, and acceptance command. Draft `docs/online-softmax.md` (per §1.4) as part of Phase 4 planning before writing the V3 kernel. ADR decisions D1–D4, D6, D7 before their phases; D5 is resolved empirically in Phase 6. Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (Phases 2, 3, 5, 6, 7)  →  /ship
```
Skip `/qa` and `/cso`. `/benchmark` after Phase 2 establishes the baseline hypothesis the whole project argues against.
